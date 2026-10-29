//! Tests of `add_rmsnorm_q8` (seam 1: the Rust wrapper against the C entry against the
//! `add_rmsnorm_q8_k` kernel). Device tests assume `--test-threads=1`.
//!
//! The kernel's group is the warnings-into-errors build WITHOUT `-use_fast_math`, so both outputs
//! are exact against the right host reference: `norm_out` against the kernel's reduction order
//! with `rsqrtf` modeled as the IEEE reciprocal square root within its documented 2 ulp (the
//! byte-exact anchor is `add_norm_pack`'s `norm_out`, same expressions in the same build group),
//! and `q8` against the host restatement of the block rule (amax over `fabsf` with `fmaxf`
//! NaN-skipping semantics, `d = amax/127`, `q = amax == 0 ? 0 : roundf(x/d)`, `ds` = the halves
//! of `d` and the butterfly sum) applied to the device-read `norm_out` — byte for byte.

use super::{AddNormQ8Call, add_rmsnorm_q8_needs};
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::resident::add_norm_pack::AddNormCall;
use crate::kernels::cuda::resident::matvec_q::tests::{Owner, assert_bad_request, reference};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// The kernel's f32 warp butterfly (xor offsets 16, 8, 4, 2, 1) over 32 lanes, in place. Shared
/// with the swiglu_q8 tests.
pub(in crate::kernels::cuda) fn butterfly32(lanes: &mut [f32; 32]) {
    let mut offset = 16;
    while offset > 0 {
        let before = *lanes;
        for lane in 0..32 {
            lanes[lane] += before[lane ^ offset];
        }
        offset /= 2;
    }
}

/// The host restatement of the producer kernels' q8_1 block rule: `amax` over
/// `fabsf` with `fmaxf` NaN-skipping semantics, `d = amax/127` (IEEE division), `q = amax == 0 ?
/// 0 : roundf(x/d)` (roundf rounds halfway cases away from zero; a NaN casts to 0 like the PTX
/// cvt), and `ds` = the halves of `d` and of the butterfly sum. The producers run in the
/// warnings-into-errors group WITHOUT `-use_fast_math`, so this is byte-exact. Shared with the
/// swiglu_q8 tests.
pub(in crate::kernels::cuda) mod q8_ref {
    use super::*;

    /// The 36 bytes of one block of 32 values.
    pub(in crate::kernels::cuda) fn block(values: &[f32; 32]) -> [u8; 36] {
        let amax = values.iter().fold(f32::NAN, |acc, v| acc.max(v.abs()));
        let d = amax / 127.0f32;
        let mut bytes = [0u8; 36];
        bytes[0..2].copy_from_slice(&testkit::f64_to_f16_bits(f64::from(d)).to_le_bytes());
        let sum = {
            let mut lanes = *values;
            butterfly32(&mut lanes);
            lanes[0]
        };
        bytes[2..4].copy_from_slice(&testkit::f64_to_f16_bits(f64::from(sum)).to_le_bytes());
        for (i, value) in values.iter().enumerate() {
            bytes[4 + i] = if amax == 0.0 {
                0
            } else {
                ((value / d).round() as i8) as u8
            };
        }
        bytes
    }

    /// The padded `block_q8_1` row of `live` values (a multiple of 32): the live blocks, then
    /// zero tail blocks up to `pad(len, 512)` values.
    pub(in crate::kernels::cuda) fn row(live: &[f32]) -> Vec<u8> {
        let padded = live.len().next_multiple_of(512);
        let mut out = Vec::with_capacity(padded / 32 * 36);
        for index in 0..padded / 32 {
            let start = index * 32;
            let mut values = [0f32; 32];
            if start < live.len() {
                values.copy_from_slice(&live[start..start + 32]);
            }
            out.extend_from_slice(&block(&values));
        }
        out
    }

    /// The row of an all-NaN input, as the device writes it: the LIVE blocks (the first
    /// `live / 32`) carry NaN `d` and `s` halves — ANY pattern with the exponent all ones and a
    /// nonzero mantissa, because the two producers disagree on the payload: the device's
    /// `__float2half(NaN)` is 0x7fff, the host oracle `f64_to_f16_bits` is 0x7e00 — with zero
    /// codes (a NaN casts to 0 like the PTX cvt), and the TAIL past the live width (the padded
    /// zeros of a width below the 512 grid, e.g. n_ff 896) is all-zero blocks: an amax of 0
    /// gives d = s = 0 and zero codes, not NaN scales.
    pub(in crate::kernels::cuda) fn assert_nan_row(bytes: &[u8], live: usize, what: &str) {
        let live_blocks = live / 32;
        let blocks = live.next_multiple_of(512) / 32;
        assert_eq!(bytes.len(), blocks * 36, "{what}: the padded row length");
        let nan_half = |at: usize| {
            let bits = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
            bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
        };
        for block in 0..blocks {
            let at = block * 36;
            if block >= live_blocks {
                assert!(
                    bytes[at..at + 36].iter().all(|&byte| byte == 0),
                    "{what}: tail block {block} is not the zero block"
                );
                continue;
            }
            assert!(
                nan_half(at) && nan_half(at + 2),
                "{what}: block {block} scales are not NaN halves"
            );
            assert!(
                bytes[at + 4..at + 36].iter().all(|&byte| byte == 0),
                "{what}: block {block} codes are not zero"
            );
        }
    }
}

/// The kernel's f32 sum of squares: thread `t` accumulates `v * v` fused (the compiled
/// `ss += v * v` contracts to one multiply-add) over `i = t, t + 256, ...`, then the block
/// reduction of `kern::block_sum`: one butterfly per warp, `red[8]`, one butterfly over the
/// zero-filled lanes 0..8.
fn reference_ss(row: &[f32]) -> f32 {
    let mut per_thread = [0f32; 256];
    for (i, value) in row.iter().enumerate() {
        let thread = i % 256;
        per_thread[thread] = value.mul_add(*value, per_thread[thread]);
    }
    let mut warp_sums = [0f32; 8];
    for warp in 0..8 {
        let mut lanes = [0f32; 32];
        lanes.copy_from_slice(&per_thread[warp * 32..warp * 32 + 32]);
        butterfly32(&mut lanes);
        warp_sums[warp] = lanes[0];
    }
    let mut lanes = [0f32; 32];
    lanes[..8].copy_from_slice(&warp_sums);
    butterfly32(&mut lanes);
    lanes[0]
}

/// The reference normalized row: the exact reciprocal square root of `ss / n + eps` (the kernel's
/// `rsqrtf` stays within its documented 2 ulp of this; the caller allows that slack) and the
/// kernel's expression order `(h * rstd) * w`.
fn reference_norm(h_after: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ss = reference_ss(h_after);
    let rstd = (1.0 / f64::sqrt(f64::from(ss / h_after.len() as f32 + eps))) as f32;
    h_after
        .iter()
        .zip(weight)
        .map(|(value, w)| (value * rstd) * w)
        .collect()
}

/// The little-endian f32 values of device bytes. Shared with the swiglu_q8 tests.
pub(in crate::kernels::cuda) fn as_f32(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// One launch of the producer under test into canaried buffers: returns the read-back `h`,
/// `norm_out` bytes and `q8` bytes.
fn launch_mine(
    ctx: &CudaContext,
    stream: &DeviceStream,
    n: usize,
    h: &[f32],
    delta: Option<&[f32]>,
    weight: &[f32],
    eps: f32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let need = add_rmsnorm_q8_needs(n).expect("sizes");
    let h_dev = testkit::upload_f32(ctx, h);
    let delta_dev = delta.map(|values| testkit::upload_f32(ctx, values));
    let weight_dev = testkit::upload_f32(ctx, weight);
    let norm = testkit::upload_filled(ctx, 0xAB, need.row + 64);
    let q8 = testkit::upload_filled(ctx, 0xA5, need.q8 + 64);
    ctx.add_rmsnorm_q8(
        stream,
        AddNormQ8Call {
            n,
            eps,
            h: h_dev.range(0, need.row).expect("h range"),
            delta: delta_dev
                .as_ref()
                .map(|buffer| buffer.range(0, need.row).expect("delta range")),
            weight: weight_dev.range(0, need.weight).expect("weight range"),
            norm_out: norm.range(0, need.row).expect("norm_out range"),
            q8: q8.range(0, need.q8).expect("q8 range"),
        },
    )
    .expect("queue add_rmsnorm_q8");
    stream.synchronize().expect("drain");
    let norm_all = testkit::read_bytes(ctx, &norm, need.row + 64);
    assert!(
        norm_all[need.row..].iter().all(|&byte| byte == 0xAB),
        "norm_out wrote past its range"
    );
    let q8_all = testkit::read_bytes(ctx, &q8, need.q8 + 64);
    assert!(
        q8_all[need.q8..].iter().all(|&byte| byte == 0xA5),
        "q8 wrote past its range"
    );
    (
        testkit::read_bytes(ctx, &h_dev, need.row),
        norm_all[..need.row].to_vec(),
        q8_all[..need.q8].to_vec(),
    )
}

/// The byte-exact anchor: `add_norm_pack` (m 1) on the same input, norm_out bytes only.
fn launch_anchor(
    ctx: &CudaContext,
    stream: &DeviceStream,
    n: usize,
    h: &[f32],
    delta: Option<&[f32]>,
    weight: &[f32],
    eps: f32,
) -> Vec<u8> {
    let row = n * 4;
    let h_dev = testkit::upload_f32(ctx, h);
    let delta_dev = delta.map(|values| testkit::upload_f32(ctx, values));
    let weight_dev = testkit::upload_f32(ctx, weight);
    let xf_need = xf_bytes(1, n).expect("xf size");
    let xf = testkit::upload_filled(ctx, 0xAA, xf_need);
    let norm = testkit::upload_filled(ctx, 0xAB, row);
    ctx.add_norm_pack(
        stream,
        AddNormCall {
            m: 1,
            n,
            eps,
            h: h_dev.range(0, row).expect("h range"),
            delta: delta_dev
                .as_ref()
                .map(|buffer| buffer.range(0, row).expect("delta range")),
            weight: weight_dev.range(0, n * 4).expect("weight range"),
            xf: xf.range(0, xf_need).expect("xf range"),
            norm_out: Some(norm.range(0, row).expect("norm_out range")),
        },
    )
    .expect("queue the add_norm_pack anchor");
    stream.synchronize().expect("drain the anchor");
    testkit::read_bytes(ctx, &norm, row)
}

#[test]
fn add_rmsnorm_q8_matches_the_host_reference() {
    if !testkit::device_present("add_rmsnorm_q8_matches_the_host_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_a17e);
    // Random rows at the three widths (Qwen 2048 and 4096, Mellum 2304 with its padded tail of
    // eight zero blocks), then the special rows at n 2048: all-zero, one outlier,
    // denormal-small.
    // (name, h, delta, weight) of one row.
    type Case = (&'static str, Vec<f32>, Option<Vec<f32>>, Vec<f32>);
    let mut cases: Vec<Case> = Vec::new();
    for n in [2048usize, 2304, 4096] {
        let with_delta = n != 2304;
        let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
        let delta = with_delta.then(|| (0..n).map(|_| (rng.normal() as f32) * 0.05).collect());
        let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
        cases.push(("random", h, delta, w));
    }
    let n = 2048;
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let delta: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    cases.push(("all-zero h", vec![0.0; n], None, w.clone()));
    let outlier = 3.7e5f32;
    let mut h = vec![0.01f32; n];
    h[1000] = outlier;
    cases.push(("outlier", h, Some(delta), w.clone()));
    // Denormal-small WITHOUT delta: with delta the add lifts the row to normal magnitude and
    // the row proves nothing denormal; alone, 4.2e-40 squares underflow to zero, so the norm is
    // rsqrtf(eps) * h * w and the q8 scales round to zero halves.
    let denormal = 4.2e-40f32;
    cases.push(("denormal-small", vec![denormal; n], None, w));
    for (name, h, delta, w) in cases {
        let (h_mine, norm_mine, q8_mine) =
            launch_mine(&ctx, &stream, h.len(), &h, delta.as_deref(), &w, eps);
        // The delta add (and only it) touched h, in IEEE: byte-exact against the host sum, and
        // without a delta h is untouched.
        let h_expected: Vec<f32> = match delta.as_deref() {
            Some(delta) => h.iter().zip(delta).map(|(value, d)| value + d).collect(),
            None => h.clone(),
        };
        let h_expected_bytes: Vec<u8> = h_expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(h_mine, h_expected_bytes, "{name}: h after the call");
        // norm_out: within rsqrtf's 2 ulp of the host reference (the FMA/no-FMA question of the
        // compiled sum folds into this slack too), and byte-identical to add_norm_pack's.
        let norm_f32 = as_f32(&norm_mine);
        let reference = reference_norm(&h_expected, &w, eps);
        let mut worst = 0.0f64;
        for (got, want) in norm_f32.iter().zip(&reference) {
            let slack = 8.0 * f32::EPSILON as f64 * want.abs() as f64 + 1e-30;
            let diff = (got - want).abs() as f64;
            assert!(diff <= slack, "{name}: norm {got} vs {want}");
            worst = worst.max(diff / (want.abs() as f64 + 1e-30));
        }
        let anchor = launch_anchor(&ctx, &stream, h.len(), &h, delta.as_deref(), &w, eps);
        assert_eq!(norm_mine, anchor, "{name}: norm_out bytes vs add_norm_pack");
        // q8: byte-exact against the host quantizer applied to the device norm row, zero
        // padding tail included (n 2304 pads to 2560: the last eight blocks are zero).
        let q8_expected = q8_ref::row(&norm_f32);
        assert_eq!(
            q8_mine, q8_expected,
            "{name}: q8 bytes vs the host quantizer"
        );
        let live_blocks = h.len() / 32;
        assert!(
            q8_mine[live_blocks * 36..].iter().all(|&byte| byte == 0),
            "{name}: the padded tail past the live blocks is not zero"
        );
        println!(
            "{name} n {}: worst norm deviation {} relative",
            h.len(),
            worst
        );
    }
}

/// The byte counts `plan_add_rmsnorm_q8` checks a call against: `n` floats for each f32 range and
/// the padded `block_q8_1` row for `q8`.
#[test]
fn add_rmsnorm_q8_needs_are_pure_and_exact() {
    let need = add_rmsnorm_q8_needs(2048).expect("sizes");
    assert_eq!(
        (need.row, need.weight, need.q8),
        (2048 * 4, 2048 * 4, 64 * 36)
    );
    // n 2304 pads to 2560 (Mellum): 80 blocks per row, not 72.
    let need = add_rmsnorm_q8_needs(2304).expect("sizes");
    assert_eq!((need.row, need.q8), (2304 * 4, 80 * 36));
    assert_eq!(add_rmsnorm_q8_needs(256).map(|n| n.q8), Some(16 * 36));
    assert_eq!(add_rmsnorm_q8_needs(4096).map(|n| n.q8), Some(128 * 36));
    assert_eq!(add_rmsnorm_q8_needs(32_768).map(|n| n.q8), Some(1024 * 36));
    assert_eq!(add_rmsnorm_q8_needs(usize::MAX), None);
    assert_eq!(add_rmsnorm_q8_needs(usize::MAX / 2), None);
}

/// The with/without-delta semantics: without `delta` nothing writes `h` and the norm is of `h`
/// alone; with `delta` `h` becomes the sum in place and the norm is of the sum, byte-identical to
/// a no-delta call on the pre-added row (the same values reach the kernel math).
#[test]
fn add_rmsnorm_q8_with_and_without_delta() {
    if !testkit::device_present("add_rmsnorm_q8_with_and_without_delta") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2048;
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_a17f);
    let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let delta: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let (h_no, norm_no, q8_no) = launch_mine(&ctx, &stream, n, &h, None, &w, eps);
    assert_eq!(
        h_no,
        h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        "without delta h is untouched"
    );
    let (h_yes, norm_yes, q8_yes) = launch_mine(&ctx, &stream, n, &h, Some(&delta), &w, eps);
    let sum: Vec<f32> = h.iter().zip(&delta).map(|(a, b)| a + b).collect();
    assert_eq!(
        h_yes,
        sum.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        "with delta h holds the sum in place"
    );
    assert_ne!(norm_no, norm_yes, "the delta changes the normalized row");
    assert_ne!(q8_no, q8_yes, "the delta changes the q8_1 row");
    let (h_pre, norm_pre, q8_pre) = launch_mine(&ctx, &stream, n, &sum, None, &w, eps);
    assert_eq!(h_pre, h_yes, "the pre-added row is untouched without delta");
    assert_eq!(
        norm_pre, norm_yes,
        "the fused norm of h + delta equals the norm of the pre-added row, byte for byte"
    );
    assert_eq!(q8_pre, q8_yes, "the q8_1 rows of the two forms are equal");
}

/// NaN semantics: a NaN row reaches the producer; `norm_out` is NaN the way `add_norm_pack_k`
/// writes it today (one NaN value poisons the sum of squares, so every norm value is NaN), the
/// q8_1 blocks carry NaN scales (d and s) with zero codes, and nothing outside the outputs moves.
#[test]
fn add_rmsnorm_q8_propagates_a_nan_row() {
    if !testkit::device_present("add_rmsnorm_q8_propagates_a_nan_row") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2048;
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_a180);
    let mut h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    h[17] = f32::NAN;
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let (h_mine, norm_mine, q8_mine) = launch_mine(&ctx, &stream, n, &h, None, &w, eps);
    assert_eq!(
        h_mine,
        h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        "no delta: h untouched"
    );
    let norm_f32 = as_f32(&norm_mine);
    assert!(
        norm_f32.iter().all(|v| v.is_nan()),
        "every norm value is NaN"
    );
    let anchor = launch_anchor(&ctx, &stream, n, &h, None, &w, eps);
    assert_eq!(
        norm_mine, anchor,
        "the NaN norm_out bytes equal add_norm_pack's on the same input"
    );
    // NaN scales: the device's __float2half(NaN) is not the host oracle's 0x7e00, so the row is
    // checked by NaN pattern (any binary16 NaN) with zero codes, not by bytes.
    q8_ref::assert_nan_row(&q8_mine, n, "the NaN row's q8_1 bytes");
}

/// One refusal case of the wrapper: the numbers of the call and where its ranges live.
#[derive(Clone, Copy)]
struct Spec {
    n: usize,
    eps: f32,
    h: (usize, usize),
    delta: Option<(usize, usize)>,
    weight: (usize, usize),
    norm_out: (usize, usize),
    q8: (usize, usize),
    norm_in: NormIn,
    q8_in: Q8In,
    delta_in: DeltaIn,
    h_of: Owner,
    stream_of: Owner,
}

/// Where `norm_out` of a case lies: its own buffer, or inside `weight`'s — an overlap case is an
/// overlap only when the two ranges share one allocation (`require_apart` compares ranges of
/// one buffer), the `YIn` pattern of the dense rig.
#[derive(Clone, Copy)]
enum NormIn {
    Own,
    Weight,
}

/// Where `q8` of a case lies: its own buffer, or inside `h`'s.
#[derive(Clone, Copy)]
enum Q8In {
    Own,
    H,
}

/// Where `delta` of a case lies: its own buffer, or on `weight`'s (the two read ranges may
/// legally share one allocation; the plan never compares them).
#[derive(Clone, Copy)]
enum DeltaIn {
    Own,
    Weight,
}

/// n 2304 (the Mellum width): every f32 row is 9216 bytes, q8 is the padded 80 blocks.
const N: usize = 2304;
const ROW: usize = N * 4;
const Q8_NEED: usize = 80 * 36;
const VALID: Spec = Spec {
    n: N,
    eps: 1e-6,
    h: (0, ROW),
    delta: Some((0, ROW)),
    weight: (0, ROW),
    norm_out: (0, ROW),
    q8: (0, Q8_NEED),
    norm_in: NormIn::Own,
    q8_in: Q8In::Own,
    delta_in: DeltaIn::Own,
    h_of: Owner::Own,
    stream_of: Owner::Own,
};

struct RefusalRig {
    stream: DeviceStream,
    h: DeviceBuffer,
    delta: DeviceBuffer,
    weight: DeviceBuffer,
    norm_out: DeviceBuffer,
    q8: DeviceBuffer,
    ctx: CudaContext,
}

/// The second adapter: a stream and an h buffer that belong to another context.
struct OtherContext {
    stream: DeviceStream,
    h: DeviceBuffer,
    _ctx: CudaContext,
}

fn attempt(rig: &RefusalRig, other: &OtherContext, spec: Spec) -> Result<(), crate::EngError> {
    let h_buffer = if matches!(spec.h_of, Owner::Other) {
        &other.h
    } else {
        &rig.h
    };
    let norm_buffer = match spec.norm_in {
        NormIn::Weight => &rig.weight,
        NormIn::Own => &rig.norm_out,
    };
    let q8_buffer = match spec.q8_in {
        Q8In::H => &rig.h,
        Q8In::Own => &rig.q8,
    };
    let delta_buffer = match spec.delta_in {
        DeltaIn::Weight => &rig.weight,
        DeltaIn::Own => &rig.delta,
    };
    let stream = if matches!(spec.stream_of, Owner::Other) {
        &other.stream
    } else {
        &rig.stream
    };
    rig.ctx.add_rmsnorm_q8(
        stream,
        AddNormQ8Call {
            n: spec.n,
            eps: spec.eps,
            h: h_buffer.range(spec.h.0, spec.h.1)?,
            delta: spec
                .delta
                .map(|(offset, bytes)| delta_buffer.range(offset, bytes))
                .transpose()?,
            weight: rig.weight.range(spec.weight.0, spec.weight.1)?,
            norm_out: norm_buffer.range(spec.norm_out.0, spec.norm_out.1)?,
            q8: q8_buffer.range(spec.q8.0, spec.q8.1)?,
        },
    )
}

/// For `plan_add_rmsnorm_q8`, every refusal is a `BadRequest` with the text of its own check,
/// nothing is queued, the backend stays usable, the valid call of the same rig runs, and a second
/// valid call with delta and weight on one buffer runs too (the plan never compares the two read
/// ranges).
#[test]
fn add_rmsnorm_q8_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("add_rmsnorm_q8_rejects_unsound_calls_before_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let rig = RefusalRig {
        stream: ctx.stream().expect("stream"),
        h: testkit::upload_filled(&ctx, 0, 16_384),
        delta: testkit::upload_filled(&ctx, 0, 16_384),
        weight: testkit::upload_filled(&ctx, 0, 16_384),
        norm_out: testkit::upload_filled(&ctx, 0xAB, 16_384),
        q8: testkit::upload_filled(&ctx, 0xA5, 8_192),
        ctx,
    };
    let other_ctx = CudaContext::new().expect("second context");
    let other = OtherContext {
        stream: other_ctx.stream().expect("stream of the other context"),
        h: testkit::upload_filled(&other_ctx, 0, 16_384),
        _ctx: other_ctx,
    };
    let cases: [(&str, &str, Spec); 18] = [
        ("n below one unit", "takes n in", Spec { n: 128, ..VALID }),
        (
            "n off the 256 grid",
            "needs n as a multiple of 256",
            Spec { n: 2368, ..VALID },
        ),
        (
            "n above the width bound",
            "takes n in",
            Spec { n: 33_024, ..VALID },
        ),
        ("eps zero", "eps finite", Spec { eps: 0.0, ..VALID }),
        (
            "eps NaN",
            "eps finite",
            Spec {
                eps: f32::NAN,
                ..VALID
            },
        ),
        (
            "h one float short",
            "h range holds",
            Spec {
                h: (0, ROW - 4),
                ..VALID
            },
        ),
        (
            "delta one float short",
            "delta range holds",
            Spec {
                delta: Some((0, ROW - 4)),
                ..VALID
            },
        ),
        (
            "weight one float short",
            "weight range holds",
            Spec {
                weight: (0, ROW - 4),
                ..VALID
            },
        ),
        (
            "norm_out one float short",
            "norm_out range holds",
            Spec {
                norm_out: (0, ROW - 4),
                ..VALID
            },
        ),
        (
            "q8 one block short",
            "q8 range holds",
            Spec {
                q8: (0, Q8_NEED - 36),
                ..VALID
            },
        ),
        (
            "q8 unpadded: 72 blocks",
            "q8 range holds",
            Spec {
                q8: (0, 72 * 36),
                ..VALID
            },
        ),
        (
            "h misaligned by 2",
            "h range starts at offset",
            Spec {
                h: (2, ROW),
                ..VALID
            },
        ),
        (
            "norm_out misaligned by 8",
            "norm_out range starts at offset",
            Spec {
                norm_out: (8, ROW),
                ..VALID
            },
        ),
        (
            "q8 misaligned by 2",
            "q8 range starts at offset",
            Spec {
                q8: (2, Q8_NEED),
                ..VALID
            },
        ),
        (
            "q8 over h",
            "h and q8 ranges overlap",
            Spec {
                q8: (4_096, Q8_NEED),
                q8_in: Q8In::H,
                ..VALID
            },
        ),
        (
            "norm_out over weight",
            "norm_out and weight ranges overlap",
            Spec {
                norm_out: (4_096, ROW),
                norm_in: NormIn::Weight,
                ..VALID
            },
        ),
        (
            "h of another context",
            "buffer belongs",
            Spec {
                h_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "a stream of another context",
            "stream belongs",
            Spec {
                stream_of: Owner::Other,
                ..VALID
            },
        ),
    ];
    for (case, needle, spec) in cases {
        assert_bad_request(attempt(&rig, &other, spec), case, needle);
    }
    // No refusal queued anything: the outputs are untouched, the backend works, and the valid
    // call runs and writes q8; the shared delta/weight form follows below.
    rig.stream.synchronize().expect("drain");
    let norm = testkit::read_bytes(&rig.ctx, &rig.norm_out, 16_384);
    assert!(norm.iter().all(|&byte| byte == 0xAB), "norm_out untouched");
    let q8 = testkit::read_bytes(&rig.ctx, &rig.q8, 8_192);
    assert!(
        q8[..Q8_NEED].iter().all(|&byte| byte == 0xA5),
        "q8 untouched"
    );
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    attempt(&rig, &other, VALID).expect("the valid call launches");
    rig.stream.synchronize().expect("drain the valid call");
    let q8 = testkit::read_bytes(&rig.ctx, &rig.q8, 8_192);
    assert!(
        q8[..Q8_NEED].iter().all(|&byte| byte != 0xA5),
        "the valid call wrote q8"
    );
    // The legality the docs claim is exercised, not just stated: a second valid call with
    // delta and weight on ONE buffer (the plan never compares the two read ranges) launches
    // and writes its outputs like the valid call above.
    attempt(
        &rig,
        &other,
        Spec {
            delta: Some((0, ROW)),
            delta_in: DeltaIn::Weight,
            ..VALID
        },
    )
    .expect("the call with delta and weight sharing one buffer launches");
    rig.stream.synchronize().expect("drain the shared call");
}

/// One stored block as (d, s, qs), the parse the matvec tests use.
fn parse_block(bytes: &[u8], at: usize) -> (f32, f32, [i8; 32]) {
    let half = |offset: usize| {
        testkit::f16_to_f32(u16::from_le_bytes([
            bytes[at + offset],
            bytes[at + offset + 1],
        ]))
    };
    let mut qs = [0i8; 32];
    for (i, q) in qs.iter_mut().enumerate() {
        *q = bytes[at + 4 + i] as i8;
    }
    (half(0), half(2), qs)
}

/// The fused producer's q8_1 row against the standalone `quantize_q8_1` on the same f32
/// `norm_out`: the vendored quantizer is built with `-use_fast_math`, so the rows are NOT
/// byte-equal; the test asserts the `quantize_q8_1` invariants (d and s within one half code of
/// the IEEE reference, codes within `d/2 + 0.07 * d` of the input) and reports the bit-equal
/// share.
#[test]
fn add_rmsnorm_q8_matches_unfused_quantize() {
    if !testkit::device_present("add_rmsnorm_q8_matches_unfused_quantize") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2048;
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_a181);
    let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let (_, norm_bytes, q8_mine) = launch_mine(&ctx, &stream, n, &h, None, &w, eps);
    let norm = as_f32(&norm_bytes);
    assert_eq!(
        q8_mine,
        q8_ref::row(&norm),
        "the fused row is the IEEE host row"
    );
    // The unfused `quantize_q8_1` of the same norm row.
    let need = add_rmsnorm_q8_needs(n).expect("sizes");
    let norm_dev = testkit::upload_bytes(&ctx, &norm_bytes);
    let unfused = testkit::upload_filled(&ctx, 0xA5, need.q8 + 64);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: norm_dev.range(0, need.row).expect("norm range"),
            y: unfused.range(0, need.q8).expect("q8 range"),
            k: n,
            ncols: 1,
        },
    )
    .expect("queue the unfused quantize");
    stream.synchronize().expect("drain");
    let unfused_all = testkit::read_bytes(&ctx, &unfused, need.q8 + 64);
    assert!(
        unfused_all[need.q8..].iter().all(|&byte| byte == 0xA5),
        "the unfused quantize wrote past its range"
    );
    let unfused = &unfused_all[..need.q8];
    let expected = q8_ref::row(&norm);
    let blocks = need.q8 / 36;
    let mut bit_equal = 0usize;
    for block in 0..blocks {
        let (d, s, qs) = parse_block(unfused, block * 36);
        let (ref_d, ref_s, ref_qs) = parse_block(&expected, block * 36);
        if unfused[block * 36..block * 36 + 36] == expected[block * 36..block * 36 + 36] {
            bit_equal += 1;
            continue;
        }
        let codes = testkit::f64_to_f16_bits(f64::from(d))
            .abs_diff(testkit::f64_to_f16_bits(f64::from(ref_d)));
        let codes_s = testkit::f64_to_f16_bits(f64::from(s))
            .abs_diff(testkit::f64_to_f16_bits(f64::from(ref_s)));
        assert!(codes <= 1, "block {block}: unfused d {d} vs {ref_d}");
        assert!(codes_s <= 1, "block {block}: unfused s {s} vs {ref_s}");
        // The 0.07 slack subsumes the half store of d as well: the dequantized q * d_half
        // carries |q| * d * 2^-11 on top of the half step, and |q| <= 127 gives at most
        // 127 * 2^-11 ~ 0.062 of d (the swiglu reference test spells the term out; the stated
        // 0.07 invariant of `quantize_q8_1` needs no extra term here).
        let eps_q = 0.07 * d.abs() + 1e-30;
        for (i, q) in qs.iter().enumerate() {
            let input = norm[block * 32 + i];
            assert!((-127..=127).contains(q), "block {block}: code {q}");
            assert!(
                (*q as f32 * d - input).abs() <= d / 2.0 + eps_q,
                "block {block} i {i}: {} * {d} vs {input}",
                *q as f32
            );
        }
        assert_eq!(ref_qs, qs, "block {block}: codes match the IEEE reference");
    }
    println!(
        "unfused bit-equal blocks: {bit_equal}/{blocks} ({} percent)",
        100 * bit_equal / blocks
    );
}

/// The end-to-end consumer check: `matvec_q` on the producer's q8_1 output equals `matvec_q` on
/// the standalone quantize of the same `norm_out`, both within the matvec tolerance
/// `64 * 2^-24 * sum(|term|)` of the f64 host reference, for a Q8_0 2048 -> 8192 projection.
#[test]
fn add_rmsnorm_q8_feeds_matvec_q() {
    if !testkit::device_present("add_rmsnorm_q8_feeds_matvec_q") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (2048, 8192);
    let ty = QuantType::Q80;
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_a182);
    let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let h: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
    let norm_w: Vec<f32> = (0..k).map(|_| 0.5 + rng.unit() as f32).collect();
    let (_, norm_bytes, q8_mine) = launch_mine(&ctx, &stream, k, &h, None, &norm_w, eps);
    let need = add_rmsnorm_q8_needs(k).expect("sizes");
    let norm_dev = testkit::upload_bytes(&ctx, &norm_bytes);
    let unfused = testkit::upload_filled(&ctx, 0xA5, need.q8);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: norm_dev.range(0, need.row).expect("norm range"),
            y: unfused.range(0, need.q8).expect("q8 range"),
            k,
            ncols: 1,
        },
    )
    .expect("queue the unfused quantize");
    stream.synchronize().expect("drain");
    let q8_unfused = testkit::read_bytes(&ctx, &unfused, need.q8);
    // Every output row against the f64 reference of ITS OWN weight row (the sibling consumer
    // tests slice w per row; one dot for all rows would compare every row to row 0).
    let (block_bytes, blocks_per_row) = (ty.block_bytes(), k / ty.block_size());
    let mut results: Vec<(String, Vec<f32>, Vec<f64>)> = Vec::new();
    for (name, bytes) in [("fused", &q8_mine), ("unfused", &q8_unfused)] {
        let x_q8 = testkit::upload_bytes(&ctx, bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, rows * 4);
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty,
                w: w.range(0, w_bytes.len()).expect("w range"),
                x_q8: x_q8.range(0, need.q8).expect("x range"),
                y: y.range(0, rows * 4).expect("y range"),
                k,
                rows,
                ncols: 1,
            },
        )
        .expect("queue the matvec");
        stream.synchronize().expect("drain");
        let y_bytes = testkit::read_bytes(&ctx, &y, rows * 4);
        let values = as_f32(&y_bytes);
        let mut tolerances = Vec::with_capacity(rows);
        let mut worst = 0.0f64;
        for (r, value) in values.iter().enumerate() {
            let w_row =
                &w_bytes[r * blocks_per_row * block_bytes..][..blocks_per_row * block_bytes];
            let (sum, abs) = reference::reference_dot(ty, w_row, bytes.as_slice(), k);
            let tol = 64.0 * f64::from(2.0f32.powi(-24)) * abs + 1e-30;
            let diff = f64::from(*value) - sum;
            assert!(
                diff.abs() <= tol,
                "{name} row {r}: {value} vs {sum} (tol {tol})"
            );
            worst = worst.max(diff.abs() / tol);
            tolerances.push(tol);
        }
        println!("{name}: worst error {} of the tolerance", worst);
        results.push((name.to_owned(), values, tolerances));
    }
    // The equality check: the two matvecs agree within the stated tolerance (each side is
    // within its own bound of the same f64 reference, so their distance is within the sum).
    let (_, fused, tol_fused) = &results[0];
    let (_, unfused_y, tol_unfused) = &results[1];
    for (((a, b), tol_a), tol_b) in fused
        .iter()
        .zip(unfused_y)
        .zip(tol_fused.iter())
        .zip(tol_unfused.iter())
    {
        assert!(
            (a - b).abs() as f64 <= tol_a + tol_b,
            "the fused and unfused matvecs differ beyond the stated tolerance: {a} vs {b}"
        );
    }
}
