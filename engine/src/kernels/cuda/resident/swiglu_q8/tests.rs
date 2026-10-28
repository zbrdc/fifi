//! Tests of `swiglu_q8` (seam 1: the Rust wrapper against the C entry against the `swiglu_q8_k`
//! kernel), and of the two producers together (`q8_producers_*`). Device tests assume
//! `--test-threads=1` (decision 0040) and run on the 4050 queue through lane L (KUBDAN-MSG-67
//! condition 6).
//!
//! silu keeps the fast `__expf` of `swiglu_pack_k` (by decision; the E2 domain of the
//! `swiglu_pack` doc), so the activation values are NOT bit-exact against an IEEE host silu:
//! the test states the tolerance (`__expf` is within `2 + floor(|1.16 * x|)` ulp, so the value
//! error is covered by `1e-5` relative) and asserts the QUANTIZATION exactly given the values
//! the kernel saw — via invariants `|q * d - y_ref| <= d/2 + 1e-5 * |y_ref| + 1e-30`, d and s
//! within one half code of the reference built from the f64 silu row — plus byte equality of
//! the zero padding tail.

/// The byte counts `plan_swiglu_q8` checks a call against (KUBDAN-MSG-64 item 6, mutant S4): the
/// tight A2 footprint of `gu` and the padded `block_q8_1` rows of `q8`.
#[test]
fn swiglu_q8_needs_are_pure_and_exact() {
    // Qwen: n_ff 512, stride 1024 (exactly 2 * n_ff), 8 slots.
    let need = swiglu_q8_needs(512, 1024, 8).expect("sizes");
    assert_eq!((need.gu, need.q8), ((7 * 1024 + 2 * 512) * 4, 8 * 16 * 36));
    // Mellum: n_ff 896 pads to 1024 (32 blocks per slot, not 28); stride above 2 * n_ff leaves
    // gaps the footprint does not include.
    let need = swiglu_q8_needs(896, 2048, 8).expect("sizes");
    assert_eq!((need.gu, need.q8), ((7 * 2048 + 2 * 896) * 4, 8 * 32 * 36));
    // The shared expert: one slot on its own buffer.
    let need = swiglu_q8_needs(512, 1024, 1).expect("sizes");
    assert_eq!((need.gu, need.q8), (2 * 512 * 4, 16 * 36));
    assert_eq!(swiglu_q8_needs(32, 64, 8).map(|n| n.q8), Some(8 * 16 * 36));
    assert!(swiglu_q8_needs(512, 1024, 0).is_none());
    assert!(swiglu_q8_needs(usize::MAX, usize::MAX, 8).is_none());
    assert!(swiglu_q8_needs(usize::MAX / 2, usize::MAX, 2).is_none());
    // A stride that WRAPS the footprint small (lane C K2 MINOR 7): two slots make the footprint
    // stride + 2 * n_ff floats, and usize::MAX - 512 + 1024 wraps to 512 — a wrapping form
    // would answer Some with a 2048-byte gu, so only the checked add refuses.
    assert!(swiglu_q8_needs(512, usize::MAX - 512, 2).is_none());
}

use super::{SwigluQ8Call, swiglu_q8_needs};
use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::{butterfly32, q8_ref};
use crate::kernels::cuda::resident::add_rmsnorm_q8::{AddNormQ8Call, add_rmsnorm_q8_needs};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceStream};

/// One block of the stored row as (d, s, qs).
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

/// The f64 silu row of one slot, rounded to f32: the fp16-free reference `silu(g) * u` the packet
/// names (an IEEE silu; the kernel's __expf differs within the stated 1e-5 relative slack).
fn reference_slot(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| (testkit::silu_f64(f64::from(*g)) * f64::from(*u)) as f32)
        .collect()
}

/// One launch of the producer into a canaried buffer; returns the read-back `q8` bytes.
fn launch_swiglu(
    ctx: &CudaContext,
    stream: &DeviceStream,
    n_ff: usize,
    stride: usize,
    slots: usize,
    gu: &[f32],
) -> Vec<u8> {
    let need = swiglu_q8_needs(n_ff, stride, slots).expect("sizes");
    assert!(
        gu.len() * 4 >= need.gu,
        "the test data covers the whole footprint"
    );
    // The A2 footprint ends at slot slots-1's up half; the stride gaps are never read, so only
    // the footprint bytes go up.
    let gu_bytes: Vec<u8> = gu.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gu_dev = testkit::upload_bytes(ctx, &gu_bytes[..need.gu]);
    let q8 = testkit::upload_filled(ctx, 0xA5, need.q8 + 64);
    ctx.swiglu_q8(
        stream,
        SwigluQ8Call {
            gu: gu_dev.range(0, need.gu).expect("gu range"),
            q8: q8.range(0, need.q8).expect("q8 range"),
            n_ff,
            y_slot_stride: stride,
            slots,
        },
    )
    .expect("queue swiglu_q8");
    stream.synchronize().expect("drain");
    let all = testkit::read_bytes(ctx, &q8, need.q8 + 64);
    assert!(
        all[need.q8..].iter().all(|&byte| byte == 0xA5),
        "q8 wrote past its range"
    );
    all[..need.q8].to_vec()
}

/// The stored row of one slot, as bytes, of the canaried buffer.
fn slot_bytes(row: &[u8], slot: usize, blocks_per_slot: usize) -> &[u8] {
    &row[slot * blocks_per_slot * 36..(slot + 1) * blocks_per_slot * 36]
}

/// The batched producer against the host reference (packet test 5): eight slots of distinct
/// data, the Qwen shape (n_ff 512, stride exactly 2 * n_ff), the Mellum shape (n_ff 896, stride
/// above 2 * n_ff, padded to 32 blocks per slot with a zero tail of four blocks) and the
/// one-slot shared-expert form. The kernel's silu keeps `__expf`, so the check is: d and s
/// within one half code of the reference row (a relative value error of 1e-5 is far below the
/// half spacing 2^-11), every code within `d/2 + 1e-5 * |y|` of the f64 reference rounded to
/// f32, codes inside [-127, 127], and the zero tail exact.
#[test]
fn swiglu_q8_matches_the_host_reference() {
    if !testkit::device_present("swiglu_q8_matches_the_host_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5eed_5187);
    let mut worst_q = 0.0f64;
    for (n_ff, stride, slots, name) in [
        (512usize, 1024usize, 8usize, "qwen n_ff 512 stride 1024"),
        (896, 2048, 8, "mellum n_ff 896 stride 2048"),
        (512, 1024, 1, "shared expert one slot"),
    ] {
        let mut gu = vec![0f32; slots * stride];
        let mut references = Vec::with_capacity(slots);
        for slot in 0..slots {
            let base = slot * stride;
            for (half, columns) in [(0, 0usize), (1, n_ff)] {
                for i in 0..n_ff {
                    gu[base + columns + i] =
                        (rng.normal() as f32) * if half == 0 { 0.5 } else { 0.25 };
                }
            }
            let gate = &gu[base..base + n_ff];
            let up = &gu[base + n_ff..base + 2 * n_ff];
            references.push(reference_slot(gate, up));
        }
        let row = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
        let blocks_per_slot = q8_1_blocks(n_ff);
        for (slot, reference) in references.iter().enumerate() {
            let stored = slot_bytes(&row, slot, blocks_per_slot);
            // The zero tail past the live blocks (n_ff 896 pads to 1024: four zero blocks).
            assert!(
                stored[n_ff / 32 * 36..].iter().all(|&byte| byte == 0),
                "{name} slot {slot}: the padded tail is not zero"
            );
            let mut amax_ref = 0.0f64;
            for block in 0..n_ff / 32 {
                let (d, s, qs) = parse_block(stored, block * 36);
                let mut ref_values = [0f32; 32];
                ref_values.copy_from_slice(&reference[block * 32..block * 32 + 32]);
                let ref_block = q8_ref::block(&ref_values);
                let (ref_d, _ref_s, _) = parse_block(&ref_block, 0);
                for value in ref_values {
                    amax_ref = amax_ref.max(f64::from(value.abs()));
                }
                let codes_d = testkit::f64_to_f16_bits(f64::from(d))
                    .abs_diff(testkit::f64_to_f16_bits(f64::from(ref_d)));
                assert!(
                    codes_d <= 1,
                    "{name} slot {slot} block {block}: d {d} vs {ref_d}"
                );
                // s: the butterfly sum of the f32 reference row, plus the __expf slack of the
                // 32 values and one half ulp of the store (2^-11 RELATIVE — the absolute value,
                // a negative sum must widen the bound, not shrink it).
                let mut lanes = ref_values;
                butterfly32(&mut lanes);
                let s_ref = f64::from(lanes[0]);
                let slack_s =
                    32.0 * 1e-5 * amax_ref + f64::from(lanes[0]).abs() * 2.0f64.powi(-11) + 1e-24;
                assert!(
                    (f64::from(s) - s_ref).abs() <= slack_s,
                    "{name} slot {slot} block {block}: s {s} vs {s_ref}"
                );
                for (i, q) in qs.iter().enumerate() {
                    let value = f64::from(ref_values[i]);
                    assert!((-127..=127).contains(q), "{name} slot {slot} code {q}");
                    // The kernel rounds x/d in f32 (d = amax/127) but STORES d as a half, so the
                    // dequantized q * d_half carries the store's rounding as well as the half
                    // step of the code: |q * d_half - x| <= d_half/2 (the roundf half step,
                    // scaled by d) + |q| * d_half * 2^-11 (the half store of d, worst |q| 127)
                    // + 1e-5 * |x| (the stated __expf slack of the value).
                    let slack = f64::from(d) * 0.5
                        + (*q as f32 as f64).abs() * f64::from(d) * 2.0f64.powi(-11)
                        + 1e-5 * value.abs()
                        + 1e-30;
                    let error = (*q as f32 as f64) * f64::from(d) - value;
                    assert!(
                        error.abs() <= slack,
                        "{name} slot {slot} block {block} i {i}: {} * {d} vs {value}",
                        *q as f32
                    );
                    worst_q = worst_q.max(error.abs() / slack);
                }
            }
        }
    }
    println!("swiglu_q8: worst code error {} of its slack", worst_q);
}

/// The number of padded blocks of one row of `n_ff` values (the q8_1 padding rule).
fn q8_1_blocks(n_ff: usize) -> usize {
    n_ff.next_multiple_of(512) / 32
}

/// Changing slot j changes only slot j's blocks (packet test 6): the other slots' rows are
/// byte-identical between the two launches.
#[test]
fn swiglu_q8_slots_are_independent() {
    if !testkit::device_present("swiglu_q8_slots_are_independent") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (n_ff, stride, slots) = (896usize, 2048usize, 8usize);
    let mut rng = testkit::Rng::new(0x5eed_5188);
    let mut gu = vec![0f32; slots * stride];
    for value in &mut gu {
        *value = (rng.normal() as f32) * 0.3;
    }
    let blocks = q8_1_blocks(n_ff);
    let baseline = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
    for changed in [0usize, 3, 7] {
        let mut edited = gu.clone();
        let base = changed * stride;
        for i in 0..n_ff {
            edited[base + i] = (rng.normal() as f32) * 2.0;
        }
        let rerun = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &edited);
        for slot in 0..slots {
            let (a, b) = (
                slot_bytes(&baseline, slot, blocks),
                slot_bytes(&rerun, slot, blocks),
            );
            if slot == changed {
                assert_ne!(a, b, "slot {slot} did not follow its changed input");
            } else {
                assert_eq!(a, b, "slot {slot} changed although only {changed} did");
            }
        }
    }
}

/// A1 semantics (KUBDAN-MSG-67 condition 3): a faulted router slot reaches the producer as a NaN
/// row (the guard writes the whole slot). Its LIVE blocks carry NaN scales — any binary16 NaN;
/// the device's `__float2half(NaN)` is 0x7fff, not the host oracle's 0x7e00 — with zero codes,
/// no crash, no clamp to zero, and no other slot's bytes and no byte outside the output moves.
/// Both shapes: the Qwen width 512 (no padded tail) and the Mellum width 896 whose four tail
/// blocks are all-zero (an amax of 0 gives d = s = 0, not NaN — lane C K2 MINOR 5).
#[test]
fn swiglu_q8_propagates_nan_slots() {
    if !testkit::device_present("swiglu_q8_propagates_nan_slots") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let slots = 8usize;
    let mut rng = testkit::Rng::new(0x5eed_5189);
    for (n_ff, stride) in [(512usize, 1024usize), (896, 2048)] {
        let mut gu = vec![0f32; slots * stride];
        for value in &mut gu {
            *value = (rng.normal() as f32) * 0.3;
        }
        let blocks = q8_1_blocks(n_ff);
        let clean = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
        let faulted_slot = 3;
        let base = faulted_slot * stride;
        for i in 0..2 * n_ff {
            gu[base + i] = f32::NAN;
        }
        let faulted = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
        for slot in 0..slots {
            let (a, b) = (
                slot_bytes(&clean, slot, blocks),
                slot_bytes(&faulted, slot, blocks),
            );
            if slot == faulted_slot {
                assert_ne!(a, b, "n_ff {n_ff}: the NaN slot produced its clean output");
                q8_ref::assert_nan_row(
                    b,
                    n_ff,
                    "n_ff {n_ff}: the NaN slot's blocks are NaN scales with zero codes",
                );
            } else {
                assert_eq!(
                    a, b,
                    "n_ff {n_ff}: slot {slot} changed although only {faulted_slot} faulted"
                );
            }
        }
    }
}

use crate::EngError;
use crate::kernels::cuda::DeviceBuffer;
use crate::kernels::cuda::resident::matvec_q::tests::{Owner, assert_bad_request};

/// One refusal case of the wrapper: the numbers of the call and where its ranges live.
#[derive(Clone, Copy)]
struct Spec {
    n_ff: usize,
    y_slot_stride: usize,
    slots: usize,
    gu: (usize, usize),
    q8: (usize, usize),
    q8_in: Q8In,
    gu_of: Owner,
    stream_of: Owner,
}

/// Where `q8` of a case lies: its own buffer, or inside `gu`'s — an overlap case is an overlap
/// only when the two ranges share one allocation (`require_apart` compares ranges of one
/// buffer), the `YIn` pattern of the dense rig.
#[derive(Clone, Copy)]
enum Q8In {
    Own,
    Gu,
}

/// The Mellum expert shape: stride 2048 above the 1792 a slot needs, so the footprint stops at
/// slot 7's up half.
const N_FF: usize = 896;
const STRIDE: usize = 2048;
const SLOTS: usize = 8;
const GU_NEED: usize = (7 * STRIDE + 2 * N_FF) * 4;
const Q8_NEED: usize = SLOTS * 32 * 36;
const VALID: Spec = Spec {
    n_ff: N_FF,
    y_slot_stride: STRIDE,
    slots: SLOTS,
    gu: (0, GU_NEED),
    q8: (0, Q8_NEED),
    q8_in: Q8In::Own,
    gu_of: Owner::Own,
    stream_of: Owner::Own,
};

struct RefusalRig {
    stream: DeviceStream,
    gu: DeviceBuffer,
    q8: DeviceBuffer,
    ctx: CudaContext,
}

/// The second adapter: a stream and a buffer that belong to another context.
struct OtherContext {
    stream: DeviceStream,
    gu: DeviceBuffer,
    _ctx: CudaContext,
}

fn attempt(rig: &RefusalRig, other: &OtherContext, spec: Spec) -> Result<(), EngError> {
    let gu_buffer = if matches!(spec.gu_of, Owner::Other) {
        &other.gu
    } else {
        &rig.gu
    };
    let stream = if matches!(spec.stream_of, Owner::Other) {
        &other.stream
    } else {
        &rig.stream
    };
    let q8_buffer = match spec.q8_in {
        Q8In::Gu => &rig.gu,
        Q8In::Own => &rig.q8,
    };
    rig.ctx.swiglu_q8(
        stream,
        SwigluQ8Call {
            gu: gu_buffer.range(spec.gu.0, spec.gu.1)?,
            q8: q8_buffer.range(spec.q8.0, spec.q8.1)?,
            n_ff: spec.n_ff,
            y_slot_stride: spec.y_slot_stride,
            slots: spec.slots,
        },
    )
}

/// KUBDAN-MSG-64 item 6 for `plan_swiglu_q8` and the shape rules of KUBDAN-MSG-67 condition 5:
/// every refusal is a `BadRequest` with the text of its own check, nothing is queued, the backend
/// stays usable, and the valid call of the same rig runs.
#[test]
fn swiglu_q8_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("swiglu_q8_rejects_unsound_calls_before_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let rig = RefusalRig {
        stream: ctx.stream().expect("stream"),
        gu: testkit::upload_filled(&ctx, 0, 65_536),
        q8: testkit::upload_filled(&ctx, 0xA5, 16_384),
        ctx,
    };
    let other_ctx = CudaContext::new().expect("second context");
    let other = OtherContext {
        stream: other_ctx.stream().expect("stream of the other context"),
        gu: testkit::upload_filled(&other_ctx, 0, 65_536),
        _ctx: other_ctx,
    };
    let cases: [(&str, &str, Spec); 16] = [
        (
            "n_ff below one block",
            "takes n_ff in",
            Spec { n_ff: 16, ..VALID },
        ),
        (
            "n_ff off the 32 grid",
            "needs n_ff as a multiple of 32",
            Spec {
                n_ff: N_FF + 16,
                ..VALID
            },
        ),
        (
            "n_ff above the width bound",
            "takes n_ff in",
            Spec {
                n_ff: 32_800,
                ..VALID
            },
        ),
        ("no slots", "takes 1 to 8 slots", Spec { slots: 0, ..VALID }),
        (
            "nine slots",
            "takes 1 to 8 slots",
            Spec { slots: 9, ..VALID },
        ),
        (
            "stride below 2*n_ff",
            "stride 1791 is below",
            Spec {
                y_slot_stride: 2 * N_FF - 1,
                ..VALID
            },
        ),
        (
            "gu one float short of the footprint",
            "gu range holds",
            Spec {
                gu: (0, GU_NEED - 4),
                ..VALID
            },
        ),
        (
            "gu packed to slots * 2*n_ff",
            "gu range holds",
            Spec {
                gu: (0, 8 * 2 * N_FF * 4),
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
            "q8 unpadded: 28 blocks per slot",
            "q8 range holds",
            Spec {
                q8: (0, 8 * 28 * 36),
                ..VALID
            },
        ),
        (
            "gu misaligned by 2",
            "gu range starts at offset",
            Spec {
                gu: (2, GU_NEED),
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
            "q8 over gu",
            "q8 and gu ranges overlap",
            Spec {
                q8: (4_096, Q8_NEED),
                q8_in: Q8In::Gu,
                ..VALID
            },
        ),
        (
            "gu of another context",
            "buffer belongs",
            Spec {
                gu_of: Owner::Other,
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
        (
            "the size overflows",
            "size overflows",
            Spec {
                y_slot_stride: usize::MAX,
                ..VALID
            },
        ),
    ];
    for (case, needle, spec) in cases {
        assert_bad_request(attempt(&rig, &other, spec), case, needle);
    }
    // No refusal queued anything: q8 is untouched, the backend works, and the valid call runs.
    rig.stream.synchronize().expect("drain");
    let q8 = testkit::read_bytes(&rig.ctx, &rig.q8, 16_384);
    assert!(
        q8[..Q8_NEED].iter().all(|&byte| byte == 0xA5),
        "q8 untouched"
    );
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    attempt(&rig, &other, VALID).expect("the valid call launches");
    rig.stream.synchronize().expect("drain the valid call");
    let q8 = testkit::read_bytes(&rig.ctx, &rig.q8, 16_384);
    assert!(
        q8[..Q8_NEED].iter().all(|&byte| byte != 0xA5),
        "the valid call wrote q8"
    );
}

/// Both producers queued inside one graph capture produce byte-identical outputs to the eager
/// launches, and the graph holds exactly their two kernel nodes (packet test 7).
#[test]
fn q8_producers_inside_a_capture_equal_eager() {
    if !testkit::device_present("q8_producers_inside_a_capture_equal_eager") {
        return;
    }
    use crate::kernels::cuda::CapturePlan;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2048;
    let (n_ff, stride, slots) = (512usize, 1024usize, 8usize);
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_ca9e);
    let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let weight: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let mut gu = vec![0f32; slots * stride];
    for value in &mut gu {
        *value = (rng.normal() as f32) * 0.3;
    }
    let k1 = add_rmsnorm_q8_needs(n).expect("sizes");
    let k2 = swiglu_q8_needs(n_ff, stride, slots).expect("sizes");
    let h_dev = testkit::upload_f32(&ctx, &h);
    let weight_dev = testkit::upload_f32(&ctx, &weight);
    let gu_dev = testkit::upload_f32(&ctx, &gu);
    let outputs = |ctx: &CudaContext| {
        (
            testkit::upload_filled(ctx, 0xAA, k1.row),
            testkit::upload_filled(ctx, 0xAA, k1.q8),
            testkit::upload_filled(ctx, 0xAA, k2.q8),
        )
    };
    let call = |ctx: &CudaContext,
                stream: &DeviceStream,
                norm: &DeviceBuffer,
                q8a: &DeviceBuffer,
                q8b: &DeviceBuffer|
     -> Result<(), EngError> {
        ctx.add_rmsnorm_q8(
            stream,
            AddNormQ8Call {
                n,
                eps,
                h: h_dev.range(0, k1.row)?,
                delta: None,
                weight: weight_dev.range(0, k1.weight)?,
                norm_out: norm.range(0, k1.row)?,
                q8: q8a.range(0, k1.q8)?,
            },
        )?;
        ctx.swiglu_q8(
            stream,
            SwigluQ8Call {
                gu: gu_dev.range(0, k2.gu)?,
                q8: q8b.range(0, k2.q8)?,
                n_ff,
                y_slot_stride: stride,
                slots,
            },
        )
    };
    let (eager_norm, eager_a, eager_b) = outputs(&ctx);
    call(&ctx, &stream, &eager_norm, &eager_a, &eager_b).expect("eager");
    stream.synchronize().expect("drain eager");
    let eager = (
        testkit::read_bytes(&ctx, &eager_norm, k1.row),
        testkit::read_bytes(&ctx, &eager_a, k1.q8),
        testkit::read_bytes(&ctx, &eager_b, k2.q8),
    );
    let (cap_norm, cap_a, cap_b) = outputs(&ctx);
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = ctx
        .capture(&stream, plan, || {
            call(&ctx, &stream, &cap_norm, &cap_a, &cap_b)
        })
        .expect("capture");
    assert_eq!(graph.nodes(), 2, "one node per producer kernel");
    for replay in 0..3 {
        for (buffer, bytes) in [(&cap_norm, k1.row), (&cap_a, k1.q8), (&cap_b, k2.q8)] {
            buffer
                .write_at(&stream, 0, &vec![0xAA; bytes])
                .expect("reset");
        }
        graph.launch(&stream).expect("replay");
        stream.synchronize().expect("drain the replay");
        assert_eq!(
            testkit::read_bytes(&ctx, &cap_norm, k1.row),
            eager.0,
            "replay {replay}: norm_out differs"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &cap_a, k1.q8),
            eager.1,
            "replay {replay}: the add_rmsnorm_q8 row differs"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &cap_b, k2.q8),
            eager.2,
            "replay {replay}: the swiglu_q8 rows differ"
        );
    }
}

/// Two launches of each producer on the same input are byte-identical (packet test 8): the fixed
/// reduction orders make a launch a pure function of its input.
#[test]
fn q8_producers_are_deterministic() {
    if !testkit::device_present("q8_producers_are_deterministic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2304;
    let (n_ff, stride, slots) = (896usize, 2048usize, 8usize);
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_ca9f);
    let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let weight: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let mut gu = vec![0f32; slots * stride];
    for value in &mut gu {
        *value = (rng.normal() as f32) * 0.3;
    }
    let k1 = add_rmsnorm_q8_needs(n).expect("sizes");
    let k2 = swiglu_q8_needs(n_ff, stride, slots).expect("sizes");
    let h_dev = testkit::upload_f32(&ctx, &h);
    let weight_dev = testkit::upload_f32(&ctx, &weight);
    let gu_dev = testkit::upload_f32(&ctx, &gu);
    let run = || -> (Vec<u8>, Vec<u8>) {
        let norm = testkit::upload_filled(&ctx, 0xAA, k1.row);
        let q8a = testkit::upload_filled(&ctx, 0xAA, k1.q8);
        let q8b = testkit::upload_filled(&ctx, 0xAA, k2.q8);
        ctx.add_rmsnorm_q8(
            &stream,
            AddNormQ8Call {
                n,
                eps,
                h: h_dev.range(0, k1.row).expect("h"),
                delta: None,
                weight: weight_dev.range(0, k1.weight).expect("w"),
                norm_out: norm.range(0, k1.row).expect("norm"),
                q8: q8a.range(0, k1.q8).expect("q8a"),
            },
        )
        .expect("norm producer");
        ctx.swiglu_q8(
            &stream,
            SwigluQ8Call {
                gu: gu_dev.range(0, k2.gu).expect("gu"),
                q8: q8b.range(0, k2.q8).expect("q8b"),
                n_ff,
                y_slot_stride: stride,
                slots,
            },
        )
        .expect("swiglu producer");
        stream.synchronize().expect("drain");
        (
            testkit::read_bytes(&ctx, &q8a, k1.q8),
            testkit::read_bytes(&ctx, &q8b, k2.q8),
        )
    };
    let first = run();
    let second = run();
    assert_eq!(first.0, second.0, "the add_rmsnorm_q8 row is deterministic");
    assert_eq!(first.1, second.1, "the swiglu_q8 rows are deterministic");
}

/// The producers write only their claimed ranges (packet test 9): the bytes past `norm_out`, `q8`
/// of both producers stay canaried, and the read-only inputs (`weight`, `gu`, and `h` without a
/// delta) hash unchanged across the launches.
#[test]
fn q8_producers_write_only_their_range() {
    if !testkit::device_present("q8_producers_write_only_their_range") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let n = 2048;
    let (n_ff, stride, slots) = (512usize, 1024usize, 8usize);
    let eps = 1e-6f32;
    let mut rng = testkit::Rng::new(0x5eed_caa0);
    let h: Vec<f32> = (0..n).map(|_| (rng.normal() as f32) * 0.05).collect();
    let weight: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
    let mut gu = vec![0f32; slots * stride];
    for value in &mut gu {
        *value = (rng.normal() as f32) * 0.3;
    }
    let k1 = add_rmsnorm_q8_needs(n).expect("sizes");
    let k2 = swiglu_q8_needs(n_ff, stride, slots).expect("sizes");
    let h_dev = testkit::upload_filled(&ctx, 0x11, k1.row + 64);
    let h_bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    h_dev.write_at(&stream, 0, &h_bytes).expect("h data");
    let weight_bytes: Vec<u8> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight_before = testkit::fnv1a64(&weight_bytes);
    let weight_dev = testkit::upload_f32(&ctx, &weight);
    let gu_bytes: Vec<u8> = gu.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gu_before = testkit::fnv1a64(&gu_bytes);
    let gu_dev = testkit::upload_f32(&ctx, &gu);
    let norm = testkit::upload_filled(&ctx, 0xAB, k1.row + 64);
    let q8a = testkit::upload_filled(&ctx, 0xA5, k1.q8 + 64);
    let q8b = testkit::upload_filled(&ctx, 0x5C, k2.q8 + 64);
    ctx.add_rmsnorm_q8(
        &stream,
        AddNormQ8Call {
            n,
            eps,
            h: h_dev.range(0, k1.row).expect("h"),
            delta: None,
            weight: weight_dev.range(0, k1.weight).expect("w"),
            norm_out: norm.range(0, k1.row).expect("norm"),
            q8: q8a.range(0, k1.q8).expect("q8a"),
        },
    )
    .expect("norm producer");
    ctx.swiglu_q8(
        &stream,
        SwigluQ8Call {
            gu: gu_dev.range(0, k2.gu).expect("gu"),
            q8: q8b.range(0, k2.q8).expect("q8b"),
            n_ff,
            y_slot_stride: stride,
            slots,
        },
    )
    .expect("swiglu producer");
    stream.synchronize().expect("drain");
    let h_back = testkit::read_bytes(&ctx, &h_dev, k1.row + 64);
    assert_eq!(&h_back[..k1.row], &h_bytes[..]);
    assert!(
        h_back[k1.row..].iter().all(|&byte| byte == 0x11),
        "h canary"
    );
    let norm_all = testkit::read_bytes(&ctx, &norm, k1.row + 64);
    assert!(
        norm_all[k1.row..].iter().all(|&byte| byte == 0xAB),
        "norm canary"
    );
    let q8a_all = testkit::read_bytes(&ctx, &q8a, k1.q8 + 64);
    assert!(
        q8a_all[k1.q8..].iter().all(|&byte| byte == 0xA5),
        "q8a canary"
    );
    let q8b_all = testkit::read_bytes(&ctx, &q8b, k2.q8 + 64);
    assert!(
        q8b_all[k2.q8..].iter().all(|&byte| byte == 0x5C),
        "q8b canary"
    );
    let weight_after = testkit::fnv1a64(&testkit::read_bytes(&ctx, &weight_dev, k1.weight));
    assert_eq!(weight_before, weight_after, "weight unchanged");
    let gu_after = testkit::fnv1a64(&testkit::read_bytes(&ctx, &gu_dev, k2.gu));
    assert_eq!(gu_before, gu_after, "gu unchanged");
}

/// Uploads little-endian i32 ids.
fn ids_buffer(ctx: &CudaContext, ids: &[i32]) -> DeviceBuffer {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    testkit::upload_bytes(ctx, &bytes)
}

/// The expert consumer (packet test 10, second half): the Q4_K expert launch on the swiglu_q8
/// output as the down-projection input (x_per_slot) equals the launch on S1's quantize of the
/// f64 reference silu rows, each within the S1 tolerance `64 * 2^-24 * sum(|term|)` of the f64
/// host dot for EVERY output of every slot.
#[test]
fn swiglu_q8_feeds_the_expert_matvec() {
    if !testkit::device_present("swiglu_q8_feeds_the_expert_matvec") {
        return;
    }
    use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::as_f32;
    use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
    use crate::kernels::cuda::resident::matvec_q::QuantType;
    use crate::kernels::cuda::resident::matvec_q::tests::reference;
    use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (n_ff, stride, slots) = (512usize, 1024usize, 8usize);
    let (k, rows, n_expert) = (n_ff, 512usize, 8usize);
    let ty = QuantType::Q4K;
    let mut rng = testkit::Rng::new(0x5eed_cc44);
    let stack_bytes = reference::build_weight(ty, rows * n_expert, k, &mut rng);
    let stack = testkit::upload_bytes(&ctx, &stack_bytes);
    let mut gu = vec![0f32; slots * stride];
    let mut ref_rows = Vec::with_capacity(slots);
    for slot in 0..slots {
        let base = slot * stride;
        for i in 0..2 * n_ff {
            gu[base + i] = (rng.normal() as f32) * 0.3;
        }
        ref_rows.push(reference_slot(
            &gu[base..base + n_ff],
            &gu[base + n_ff..base + 2 * n_ff],
        ));
    }
    // Path A: the producer under test. Path B: S1's quantize of the reference silu rows.
    let q8_mine = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
    let flat_ref: Vec<f32> = ref_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let ref_dev = testkit::upload_f32(&ctx, &flat_ref);
    let unfused_bytes = slots * q8_1_blocks(n_ff) * 36;
    let unfused = testkit::upload_filled(&ctx, 0xA5, unfused_bytes);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: ref_dev.range(0, flat_ref.len() * 4).expect("ref rows"),
            y: unfused.range(0, unfused_bytes).expect("q8"),
            k,
            ncols: slots,
        },
    )
    .expect("unfused quantize");
    stream.synchronize().expect("drain");
    let q8_unfused = testkit::read_bytes(&ctx, &unfused, unfused_bytes);
    let ids = ids_buffer(&ctx, &(0..slots as i32).collect::<Vec<_>>());
    let y_bytes = rows * slots * 4;
    let row_bytes = q8_1_blocks(n_ff) * 36;
    let mut outputs = Vec::new();
    for (name, bytes) in [("producer", &q8_mine), ("unfused", &q8_unfused)] {
        let x_q8 = testkit::upload_bytes(&ctx, bytes);
        let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
        ctx.matvec_expert(
            &stream,
            ExpertMatvecCall {
                ty,
                w_stack: stack.range(0, stack_bytes.len()).expect("stack"),
                ids: ids.range(0, slots * 4).expect("ids"),
                x_q8: x_q8.range(0, unfused_bytes).expect("x"),
                x_per_slot: true,
                y: y.range(0, y_bytes).expect("y"),
                y_slot_stride: rows,
                k,
                rows,
                n_expert,
                n_used: slots,
            },
        )
        .expect("expert matvec");
        stream.synchronize().expect("drain");
        outputs.push((name, as_f32(&testkit::read_bytes(&ctx, &y, y_bytes))));
    }
    // Every output of both paths against the f64 reference of its own q8_1 input.
    let block_bytes = ty.block_bytes();
    let blocks_per_row = k / ty.block_size();
    let mut tolerances = Vec::with_capacity(slots * rows);
    let mut abs_max = 0.0f64;
    for (name, values) in &outputs {
        let bytes = if *name == "producer" {
            &q8_mine
        } else {
            &q8_unfused
        };
        for slot in 0..slots {
            let x_row = &bytes[slot * row_bytes..(slot + 1) * row_bytes];
            for r in 0..rows {
                let w_row = &stack_bytes[(slot * rows + r) * blocks_per_row * block_bytes..]
                    [..blocks_per_row * block_bytes];
                let (sum, abs) = reference::reference_dot(ty, w_row, x_row, k);
                abs_max = abs_max.max(abs);
                let tol = 64.0 * f64::from(2.0f32.powi(-24)) * abs + 1e-30;
                let got = f64::from(values[slot * rows + r]);
                assert!(
                    (got - sum).abs() <= tol,
                    "{name} slot {slot} row {r}: {got} vs {sum} (tol {tol})"
                );
                tolerances.push(tol);
            }
        }
    }
    // And the two paths agree within the sum of their bounds PLUS one legitimate quantization
    // flip (lane C K2 MINOR 4): the two paths quantize values that differ by __expf's slack, so
    // their rows are byte-equal only while every amax and every roundf boundary land the same
    // way on this SM; on another SM a single block of one row can legitimately differ — a d
    // half-code moves that block's terms by 2^-11 relative (bounded by one block's share,
    // 2^-11 * abs_max / blocks, of the largest row's |term| sum) and a q code flip moves one
    // term by d * |w| (bounded by the largest stored scale times the builder's largest Q4_K
    // value, d 2.0 x scale 63 x nibble 15 + dmin 0.21 x 63 ~ 1.9e3). Lane C estimated a ~5
    // percent flake chance on another SM without the allowance; both bounds are explicit.
    let d_max = [&q8_mine, &q8_unfused]
        .iter()
        .flat_map(|bytes| (0..slots * q8_1_blocks(n_ff)).map(|b| parse_block(bytes, b * 36).0))
        .map(f64::from)
        .fold(0.0_f64, f64::max);
    let one_flip = 2.0f64.powi(-11) * abs_max / (k / 32) as f64 + d_max * 1_903.0;
    let (_, a) = &outputs[0];
    let (_, b) = &outputs[1];
    for (((a, b), tol_a), tol_b) in a
        .iter()
        .zip(b)
        .zip(tolerances.iter())
        .zip(tolerances[slots * rows..].iter())
    {
        assert!(
            (a - b).abs() as f64 <= tol_a + tol_b + one_flip,
            "the producer and unfused expert matvecs differ beyond one flip: {a} vs {b}"
        );
    }
}

/// The Mellum consumer: Q8_0 and Q5_0 down projections at K 896 read the producer's padded rows
/// (32 blocks, the zero tail included) and match S1's quantize of the reference silu rows within
/// the S1 tolerance for every output row (packet, Mellum shapes section).
#[test]
fn swiglu_q8_feeds_the_dense_matvec_at_896() {
    if !testkit::device_present("swiglu_q8_feeds_the_dense_matvec_at_896") {
        return;
    }
    use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::as_f32;
    use crate::kernels::cuda::resident::matvec_q::tests::reference;
    use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
    use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (n_ff, stride, slots) = (896usize, 2048usize, 8usize);
    let k = n_ff;
    let rows = 2048usize;
    let mut rng = testkit::Rng::new(0x5eed_cc45);
    let mut gu = vec![0f32; slots * stride];
    let mut ref_rows = Vec::with_capacity(slots);
    for slot in 0..slots {
        let base = slot * stride;
        for i in 0..2 * n_ff {
            gu[base + i] = (rng.normal() as f32) * 0.3;
        }
        ref_rows.push(reference_slot(
            &gu[base..base + n_ff],
            &gu[base + n_ff..base + 2 * n_ff],
        ));
    }
    let q8_mine = launch_swiglu(&ctx, &stream, n_ff, stride, slots, &gu);
    // S1's quantize of the reference rows, one column per slot.
    let flat_ref: Vec<f32> = ref_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let ref_dev = testkit::upload_f32(&ctx, &flat_ref);
    let row_bytes = q8_1_blocks(n_ff) * 36;
    let unfused_bytes = slots * row_bytes;
    let unfused = testkit::upload_filled(&ctx, 0xA5, unfused_bytes);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: ref_dev.range(0, flat_ref.len() * 4).expect("ref rows"),
            y: unfused.range(0, unfused_bytes).expect("q8"),
            k,
            ncols: slots,
        },
    )
    .expect("unfused quantize");
    stream.synchronize().expect("drain");
    let q8_unfused = testkit::read_bytes(&ctx, &unfused, unfused_bytes);
    for ty in [QuantType::Q80, QuantType::Q50] {
        let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let (block_bytes, blocks_per_row) = (ty.block_bytes(), k / ty.block_size());
        for slot in [0usize, 5] {
            for (name, bytes) in [("producer", &q8_mine), ("unfused", &q8_unfused)] {
                let x_row = &bytes[slot * row_bytes..(slot + 1) * row_bytes];
                let x_q8 = testkit::upload_bytes(&ctx, x_row);
                let y = testkit::upload_filled(&ctx, 0xAA, rows * 4);
                ctx.matvec_q(
                    &stream,
                    MatvecCall {
                        ty,
                        w: w.range(0, w_bytes.len()).expect("w"),
                        x_q8: x_q8.range(0, row_bytes).expect("x"),
                        y: y.range(0, rows * 4).expect("y"),
                        k,
                        rows,
                        ncols: 1,
                    },
                )
                .expect("dense matvec");
                stream.synchronize().expect("drain");
                let values = as_f32(&testkit::read_bytes(&ctx, &y, rows * 4));
                let mut worst = 0.0f64;
                for (r, value) in values.iter().enumerate() {
                    let w_row = &w_bytes[r * blocks_per_row * block_bytes..]
                        [..blocks_per_row * block_bytes];
                    let (sum, abs) = reference::reference_dot(ty, w_row, x_row, k);
                    let tol = 64.0 * f64::from(2.0f32.powi(-24)) * abs + 1e-30;
                    assert!(
                        (f64::from(*value) - sum).abs() <= tol,
                        "{ty:?} {name} slot {slot} row {r}: {value} vs {sum} (tol {tol})"
                    );
                    worst = worst.max((f64::from(*value) - sum).abs() / tol);
                }
                println!(
                    "{ty:?} {name} slot {slot}: worst {} of the tolerance",
                    worst
                );
            }
        }
    }
}
