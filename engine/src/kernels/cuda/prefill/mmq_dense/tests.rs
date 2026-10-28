//! Tests of the dense MMQ entry (CPF-V4 G2, seam 1). Host rows: the pure rule pins and the
//! refusal tables (one case per guard, each passing every earlier check so it reaches its
//! own tag, the sisters' rule), the size pins; device rows follow below.

use super::{DenseNeeds, MmqDenseCall, MmqDenseInputCall, check_dense_rules, dense_needs};
use crate::EngError;
use crate::kernels::cuda::native::prefill::mmq_dense::{MmqDenseRaw, mmq_dense_launch};
use crate::kernels::cuda::prefill::mmq_quant::MmqQuantizeCall;
use crate::kernels::cuda::prefill::mmq_size::{
    MmqType, NCOLS_LIMIT, Q8_1_PAD, fixup_floats_exact, fixup_floats_max, j_best, j_ub,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::Rng;
use crate::kernels::cuda::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceStream, NativeWeight, QuantTensor, testkit,
};
use crate::weights::{GgufType, dequantize};

/// The error text of a refusal, so the tables need no `Debug` on the `Ok` side.
fn refusal(result: Result<super::DenseInts, crate::EngError>) -> String {
    match result {
        Ok(ints) => panic!("expected a refusal, got {ints:?}"),
        Err(error) => format!("{error:?}"),
    }
}

#[test]
fn mmq_dense_rule_pins_convert_the_integers() {
    // The three dense types at real Qwen shapes, every converted integer checked.
    for (ty, ggml) in [
        (QuantType::Q80, 8),
        (QuantType::Q4K, 12),
        (QuantType::Q6K, 14),
    ] {
        let ints = check_dense_rules(ty, 2048, 8192, 49, 8192, 0).expect("the shape is legal");
        assert_eq!(
            (ints.ty, ints.k, ints.n, ints.m, ints.ld, ints.col_off),
            (ggml, 2048, 8192, 49, 8192, 0)
        );
    }
    // A slice of a fused destination: ld wider than the slice, a column offset.
    let fused = check_dense_rules(QuantType::Q80, 4096, 2048, 1000, 6144, 2048).expect("fused");
    assert_eq!(
        (fused.k, fused.n, fused.m, fused.ld, fused.col_off),
        (4096, 2048, 1000, 6144, 2048)
    );
}

#[test]
fn mmq_dense_refuses_a_type_without_a_case_by_tag() {
    let text = refusal(check_dense_rules(QuantType::Q50, 2048, 512, 8, 512, 0));
    assert!(text.contains("mmq_dense: type"), "{text}");
}

#[test]
fn mmq_dense_refuses_the_k_grid_by_tag() {
    // Rev 3's K-grid rule (MSG-85 M2): K = 384 is a multiple of 128 and of every flat qk
    // but not of 256; it must reach THIS tag for every type, Q4_K and Q6_K included, and
    // before the k extent rule (33,000 is off the grid AND above the extent).
    for ty in [QuantType::Q80, QuantType::Q4K, QuantType::Q6K] {
        for k in [384usize, 260, 130, 33_000] {
            let text = refusal(check_dense_rules(ty, k, 512, 8, 512, 0));
            assert!(text.contains("mmq_dense: K-grid"), "{ty:?} {k}: {text}");
        }
        for k in [256usize, 512, 2048, 4096, 32_768] {
            check_dense_rules(ty, k, 512, 8, 512, 0).expect("on the k grid");
        }
    }
}

#[test]
fn mmq_dense_refuses_the_k_extent_by_tag() {
    // 0 and 33,024 are both on the 256 grid, so only the extent rule can refuse them.
    for k in [0usize, 33_024] {
        let text = refusal(check_dense_rules(QuantType::Q80, k, 512, 8, 512, 0));
        assert!(text.contains("mmq_dense: K extent"), "{k}: {text}");
    }
}

#[test]
fn mmq_dense_pins_the_chunk_cap() {
    // G1's domain is 1..=2,048 columns; raising the chunk cap without extending it must
    // fail here (plan section B.3, finding 22).
    assert_eq!(NCOLS_LIMIT, 2048);
    for m in [0usize, 2049, 4096] {
        let text = refusal(check_dense_rules(QuantType::Q80, 2048, 512, m, 512, 0));
        assert!(text.contains("mmq_dense: M"), "{m}: {text}");
    }
    for m in [1usize, 2048] {
        check_dense_rules(QuantType::Q80, 2048, 512, m, 512, 0).expect("inside the cap");
    }
}

#[test]
fn mmq_dense_refuses_a_row_count_of_zero_by_tag() {
    let text = refusal(check_dense_rules(QuantType::Q4K, 2048, 0, 8, 0, 0));
    assert!(text.contains("mmq_dense: N"), "{text}");
}

#[test]
fn mmq_dense_refuses_a_short_destination_stride_by_tag() {
    let (n, off) = (512usize, 64usize);
    let text = refusal(check_dense_rules(
        QuantType::Q80,
        2048,
        n,
        8,
        n + off - 1,
        off,
    ));
    assert!(text.contains("mmq_dense: ld"), "{text}");
    check_dense_rules(QuantType::Q80, 2048, n, 8, n + off, off).expect("the exact stride");
    // col_off + n wraps usize: the checked add must refuse it, not wrap to a small reach.
    let text = refusal(check_dense_rules(
        QuantType::Q80,
        2048,
        2,
        8,
        usize::MAX,
        usize::MAX,
    ));
    assert!(text.contains("mmq_dense: ld"), "{text}");
}

#[test]
fn mmq_dense_mirrors_the_grid_limit_at_the_smallest_tile() {
    // ceil(m/8) * ceil(n/128) * (k/qk) < 2^30 (mmq.cuh:1402), bounded with J = 8. Q8_0 at
    // K = 32,768 has 1,024 blocks per row; one column tile (m = 8) and 2^20 row tiles
    // reach the limit exactly, one row tile fewer stays below it.
    let tiles = 1usize << 20;
    let at = |row_tiles: usize| {
        let n = row_tiles * 128;
        check_dense_rules(QuantType::Q80, 32_768, n, 8, n, 0)
    };
    let text = refusal(at(tiles));
    assert!(text.contains("mmq_dense: grid limit"), "{text}");
    at(tiles - 1).expect("one row tile below the limit");
    // The product itself overflowing usize is a refusal too, not a wrap to a small grid.
    let text = refusal(check_dense_rules(
        QuantType::Q80,
        256,
        usize::MAX,
        2048,
        usize::MAX,
        0,
    ));
    assert!(text.contains("mmq_dense: grid limit"), "{text}");
}

#[test]
fn mmq_dense_refuses_a_row_count_above_i32_by_tag() {
    // 2^31 rows of Q4_K at K = 256 and one column tile: the grid (2^24) is legal, so only
    // the i32 conversion of the C ABI can refuse it.
    let n = 1usize << 31;
    let text = refusal(check_dense_rules(QuantType::Q4K, 256, n, 8, n, 0));
    assert!(text.contains("mmq_dense: n: n is"), "{text}");
}

#[test]
fn mmq_dense_refuses_a_destination_reach_beyond_i32_by_tag() {
    // Verdict M-1: the vendored kernels index the destination in 32-bit int, so the last
    // element, at (m - 1) * ld + n - 1, must stay below 2^31. M = 2,048, ld = 2^20 and
    // n = 2^20 - 1 reach exactly i32::MAX; one more row wraps. Everything before the rule
    // passes (the grid is 16.8 M), so only the reach can refuse.
    let at = |m: usize, n: usize, ld: usize| check_dense_rules(QuantType::Q80, 256, n, m, ld, 0);
    at(2048, (1 << 20) - 1, 1 << 20).expect("a reach of exactly i32::MAX");
    let text = refusal(at(2048, 1 << 20, 1 << 20));
    assert!(text.contains("mmq_dense: dst reach"), "{text}");
    // The verdict's example, which every other rule accepted before: 1,920 * 1,118,482
    // wraps in the last column tile.
    let text = refusal(at(2048, 128, 1_118_482));
    assert!(text.contains("mmq_dense: dst reach"), "{text}");
    // One column never reaches past n, however wide the stride is.
    at(1, 128, i32::MAX as usize).expect("one column: the reach is n");
}

#[test]
fn mmq_dense_needs_pin_the_three_ranges() {
    // Weight blocks, the q8_1 columns with G1's constant pad, the destination reach.
    let need = dense_needs(QuantType::Q80, 2048, 8192, 512, 8192, 0).expect("needs");
    assert_eq!(
        need,
        DenseNeeds {
            w: 8192 * 64 * 34,
            y: 16 * 512 * 144 + Q8_1_PAD,
            dst: 512 * 8192 * 4,
        }
    );
    let need = dense_needs(QuantType::Q4K, 4096, 2048, 1000, 6144, 2048).expect("fused needs");
    assert_eq!(
        need,
        DenseNeeds {
            w: 2048 * 16 * 144,
            y: 32 * 1000 * 144 + Q8_1_PAD,
            dst: (999 * 6144 + 2048 + 2048) * 4,
        }
    );
}

#[test]
fn mmq_dense_needs_refuse_zero_columns_without_panicking() {
    // `dense_needs` is `pub(crate)`: a caller that skipped the rules and passed m = 0 gets a
    // typed refusal, not a debug panic or a wrapped reach (verdict n-4).
    let error = dense_needs(QuantType::Q80, 2048, 512, 0, 512, 0).expect_err("no columns");
    assert!(
        format!("{error:?}").contains("mmq_dense: size overflow"),
        "{error:?}"
    );
}

#[test]
fn mmq_dense_needs_refuse_overflow_by_tag() {
    for (n, m, ld) in [(usize::MAX / 2, 8, 512), (512, 4, usize::MAX / 2)] {
        let error = dense_needs(QuantType::Q80, 2048, n, m, ld, 0).expect_err("overflow");
        assert!(format!("{error:?}").contains("size overflow"), "{error:?}");
    }
}

// ---- the fixup size ------------------------------------------------------------------------

#[test]
fn mmq_fixup_bytes_follow_the_device_sm_count() {
    // G4's facts binding owns the cc / nsm / smpbo device test; this row pins what G2 builds
    // on it: the fixup range an arena allocates is G1's `nsm * 128 * 128` floats for the SM
    // count the device reports, never a literal.
    const NAME: &str = "mmq_fixup_bytes_follow_the_device_sm_count";
    if !testkit::device_present(NAME) {
        return;
    }
    // Result-only phase: the context is dropped before any assert.
    let phase = || -> Result<(usize, usize), EngError> {
        let ctx = CudaContext::new()?;
        Ok((facts_of(&ctx)?.nsm, ctx.mmq_fixup_bytes()?))
    };
    let (nsm, fixup_bytes) = phase().expect("the facts phase");
    println!("MMQ fixup: nsm {nsm}, fixup {fixup_bytes} B");
    assert!(nsm >= 1, "{nsm} SMs");
    assert_eq!(
        fixup_bytes,
        fixup_floats_max(nsm).expect("the fixup size") * 4
    );
}

// ---- the device fixture --------------------------------------------------------------------

/// The poison of an unwritten byte: as an f32 it is NaN, and as a q8_1 scale word too, so the
/// finiteness gate catches a skipped store before any bound check.
const POISON: u8 = 0xff;

fn gguf_of(ty: QuantType) -> GgufType {
    match ty {
        QuantType::Q80 => GgufType::Q80,
        QuantType::Q50 => GgufType::Q50,
        QuantType::Q4K => GgufType::Q4K,
        QuantType::Q6K => GgufType::Q6K,
    }
}

/// Host-built Q8_0 blocks of an `n` x `k` weight: Gaussian rows quantized per 32 values the
/// way ggml does (scale `amax / 127` rounded to f16, round-to-nearest codes), so the blocks
/// carry a real weight's code distribution.
fn q8_0_weight(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n * (k / 32) * 34);
    for _ in 0..n * (k / 32) {
        let values: Vec<f64> = (0..32).map(|_| rng.normal() * 0.02).collect();
        let amax = values.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let scale_bits = testkit::f64_to_f16_bits(amax / 127.0);
        let scale = f64::from(testkit::f16_to_f32(scale_bits));
        out.extend_from_slice(&scale_bits.to_le_bytes());
        for v in values {
            out.push((v / scale).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

/// `rows` activation rows of `k` finite f32: Gaussian with one value in 64 scaled by 6, so
/// the 32-value scale groups see LLM-like outliers.
fn activations(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * k)
        .map(|_| {
            let boost = if rng.below(64) == 0 { 6.0 } else { 1.0 };
            (rng.normal() * boost) as f32
        })
        .collect()
}

/// A Native weight of exactly the bytes given, uploaded in whole-row chunks.
fn upload_weight(
    ctx: &CudaContext,
    ty: QuantType,
    shape: (usize, usize),
    bytes: &[u8],
) -> Result<NativeWeight, EngError> {
    let tensor = QuantTensor {
        name: "mmq dense test weight",
        ty: gguf_of(ty),
        n_rows: shape.0,
        k: shape.1,
        source_len: bytes.len(),
    };
    ctx.upload_native(tensor, CHUNK_LIMIT, &mut |at, buf| {
        buf.copy_from_slice(&bytes[at..at + buf.len()]);
        Ok(())
    })
}

/// What a refusal-table phase hands back: the control launch, one error text per case, and
/// the three shared buffers before and after the cases.
struct Refusals {
    control: Result<(), EngError>,
    cases: Vec<(&'static str, Result<(), EngError>)>,
    repeat: Result<(), EngError>,
    before: [Vec<u8>; 3],
    after: [Vec<u8>; 3],
}

#[test]
fn mmq_dense_wrapper_refuses_bad_ranges_by_tag_and_changes_nothing() {
    // The B1 row of plan section B.5 item 8 (a q8_1 range one byte short of `q8_1_bytes`
    // with the constant pad) and the other range guards of the wrapper, each case a valid
    // control with exactly one range changed so it reaches its own guard.
    const NAME: &str = "mmq_dense_wrapper_refuses_bad_ranges_by_tag_and_changes_nothing";
    if !testkit::device_present(NAME) {
        return;
    }
    let (k, n, m) = (256usize, 128usize, 8usize);
    let (w_bytes, x) = (
        q8_0_weight(n, k, 0x6432_0001),
        activations(m, k, 0x6432_0002),
    );
    let phase = || -> Result<Refusals, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let weight = upload_weight(&ctx, QuantType::Q80, (n, k), &w_bytes)?;
        let need = dense_needs(QuantType::Q80, k, n, m, n, 0)?;
        let floor = ctx.mmq_fixup_bytes()?;
        let xb = testkit::upload_f32(&ctx, &x);
        let y = testkit::upload_filled(&ctx, POISON, need.y + need.dst);
        let dst = testkit::upload_filled(&ctx, POISON, need.dst + 64);
        let fixup = testkit::upload_filled(&ctx, 0, floor + 64);
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q80,
                k,
                row_stride: k,
                ncols: m,
                x: xb.range(0, xb.bytes())?,
                gather: None,
                y: y.range(0, need.y)?,
            },
        )?;
        let call = |q8_1, dst, fixup| MmqDenseCall {
            weight: &weight,
            w_seg: None,
            m,
            q8_1,
            dst,
            ld: n,
            col_off: 0,
            fixup,
        };
        let control = ctx.mmq_dense(
            &stream,
            call(
                y.range(0, need.y)?,
                dst.range(0, need.dst)?,
                fixup.range(0, floor)?,
            ),
        );
        stream.synchronize()?;
        let snapshot = |buffer: &DeviceBuffer| testkit::read_bytes(&ctx, buffer, buffer.bytes());
        let before = [snapshot(&y), snapshot(&dst), snapshot(&fixup)];
        let (yr, dr, fr) = (
            y.range(0, need.y)?,
            dst.range(0, need.dst)?,
            fixup.range(0, floor)?,
        );
        let run = |name, q8_1, dst, fixup| (name, ctx.mmq_dense(&stream, call(q8_1, dst, fixup)));
        let cases = vec![
            run("q8_1 one byte short", y.range(0, need.y - 1)?, dr, fr),
            run("q8_1 misaligned", y.range(4, need.y)?, dr, fr),
            run("dst one float short", yr, dst.range(0, need.dst - 4)?, fr),
            run("dst misaligned", yr, dst.range(2, need.dst)?, fr),
            run("fixup below the floor", yr, dr, fixup.range(0, floor - 16)?),
            run("fixup misaligned", yr, dr, fixup.range(4, floor)?),
            run("dst overlaps q8_1", yr, y.range(need.y - 64, need.dst)?, fr),
            run("fixup overlaps dst", yr, fixup.range(1024, need.dst)?, fr),
        ];
        // Taken before the repeat launch: that launch rewrites dst and the fixup range with the
        // control's bytes and would mask a write by a refused case into them.
        let after = [snapshot(&y), snapshot(&dst), snapshot(&fixup)];
        let repeat = ctx.mmq_dense(&stream, call(yr, dr, fr));
        stream.synchronize()?;
        Ok(Refusals {
            control,
            cases,
            repeat,
            before,
            after,
        })
    };
    let out = phase().expect("the refusal phase");
    out.control.as_ref().expect("the valid control launches");
    let expect = [
        ("the CUDA q8_1 range holds", "BadRequest"),
        (
            "the CUDA q8_1 range starts at offset 4, which is not a multiple of 16",
            "BadRequest",
        ),
        ("the CUDA dst range holds", "BadRequest"),
        (
            "the CUDA dst range starts at offset 2, which is not a multiple of 4",
            "BadRequest",
        ),
        ("the CUDA fixup range holds", "BadRequest"),
        (
            "the CUDA fixup range starts at offset 4, which is not a multiple of 16",
            "BadRequest",
        ),
        ("the CUDA dst and q8_1 ranges overlap", "BadRequest"),
        ("the CUDA fixup and dst ranges overlap", "BadRequest"),
    ];
    assert_eq!(out.cases.len(), expect.len());
    for ((name, result), (needle, kind)) in out.cases.iter().zip(expect) {
        let text = format!("{:?}", result.as_ref().expect_err(name));
        assert!(
            text.starts_with(kind) && text.contains(needle),
            "{name}: {text}"
        );
    }
    // A refused call changes nothing and does not latch the backend: the repeat launch ran
    // and every buffer is byte-for-byte what the control left.
    out.repeat
        .as_ref()
        .expect("the repeat launch after the refusals");
    assert_eq!(out.before, out.after, "a refused call must not write");
}

fn mmq_type_of(ty: QuantType) -> MmqType {
    match ty {
        QuantType::Q80 => MmqType::Q80,
        QuantType::Q50 => MmqType::Q50,
        QuantType::Q4K => MmqType::Q4K,
        QuantType::Q6K => MmqType::Q6K,
    }
}

/// Native blocks of `ty`: realistic Q8_0 codes, the testkit's finite random blocks for the
/// K-quants (scale fields set, so every dequantized value is finite).
fn weight_bytes(ty: QuantType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    match ty {
        QuantType::Q80 => q8_0_weight(n, k, seed),
        other => testkit::weight_source(gguf_of(other), n, k),
    }
}

// ---- the dense GEMM rows (Q8X, the fused slice, determinism) --------------------------------

/// The frozen Q8X bound for dense Q8_0 (0049 A2): per-case energy-weighted NMSE, the sum of
/// squared errors over the sum of squared references. Measured worst on real activations:
/// 3.15e-4 over 2,340 cases (CPF-M1b section 9).
const Q8X_DENSE_Q80: f64 = 5e-4;
/// Bytes past every output range, poisoned the same way.
const CANARY: usize = 64;

#[derive(Clone)]
struct Case {
    name: &'static str,
    ty: QuantType,
    k: usize,
    n: usize,
    m: usize,
    ld: usize,
    col_off: usize,
}

/// One case with its host weight blocks and activation rows.
struct CaseInput {
    case: Case,
    w: Vec<u8>,
    x: Vec<f32>,
}

/// The whole destination (`m * ld` floats plus the canary tail) and the whole q8_1 range as
/// the GEMM left them.
struct Run {
    dst: Vec<u8>,
    y: Vec<u8>,
    fixup_tail: Vec<u8>,
}

/// A fixup buffer of `bytes` zeros (initialized, as the arena's one zero fill makes it)
/// followed by a poisoned canary tail that no launch may write.
fn fixup_buffer(ctx: &CudaContext, bytes: usize) -> DeviceBuffer {
    let mut host = vec![0u8; bytes];
    host.extend([POISON; CANARY]);
    testkit::upload_bytes(ctx, &host)
}

/// The canary tail after the first `bytes` of a fixup buffer.
fn fixup_tail(ctx: &CudaContext, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
    testkit::read_bytes(ctx, buffer, bytes + CANARY).split_off(bytes)
}

/// The device objects a phase shares; field order is the drop order.
struct Rig {
    fixup: DeviceBuffer,
    fixup_bytes: usize,
    stream: DeviceStream,
    ctx: CudaContext,
}

fn open_rig() -> Result<Rig, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let fixup_bytes = ctx.mmq_fixup_bytes()?;
    let fixup = fixup_buffer(&ctx, fixup_bytes);
    Ok(Rig {
        fixup,
        fixup_bytes,
        stream,
        ctx,
    })
}

/// Quantizes `x`, runs the dense GEMM into a poisoned destination and reads both back.
fn run_case(rig: &Rig, case: &Case, weight: &NativeWeight, x: &[f32]) -> Result<Run, EngError> {
    let Rig {
        ctx,
        stream,
        fixup,
        fixup_bytes,
    } = rig;
    let need = dense_needs(case.ty, case.k, case.n, case.m, case.ld, case.col_off)?;
    let xb = testkit::upload_f32(ctx, x);
    // The q8_1 range is its own exact-size allocation, so a sanitizer sees every read past it.
    let y = testkit::upload_filled(ctx, POISON, need.y);
    ctx.mmq_dense_input(
        stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
            weight,
            m: case.m,
            row_stride: case.k,
            x: xb.range(0, xb.bytes())?,
            y: y.range(0, need.y)?,
        },
    )?;
    let total = case.m * case.ld * 4;
    let dst = testkit::upload_filled(ctx, POISON, total + CANARY);
    ctx.mmq_dense(
        stream,
        MmqDenseCall {
            weight,
            w_seg: None,
            m: case.m,
            q8_1: y.range(0, need.y)?,
            dst: dst.range(0, need.dst)?,
            ld: case.ld,
            col_off: case.col_off,
            fixup: fixup.range(0, *fixup_bytes)?,
        },
    )?;
    stream.synchronize()?;
    Ok(Run {
        dst: testkit::read_bytes(ctx, &dst, total + CANARY),
        y: testkit::read_bytes(ctx, &y, need.y),
        fixup_tail: fixup_tail(ctx, fixup, *fixup_bytes),
    })
}

/// The device phase of a table of cases: Result-only, every owner dropped on return.
fn run_cases(inputs: &[CaseInput]) -> Result<Vec<Run>, EngError> {
    let rig = open_rig()?;
    inputs
        .iter()
        .map(|input| {
            let case = &input.case;
            let weight = upload_weight(&rig.ctx, case.ty, (case.n, case.k), &input.w)?;
            run_case(&rig, case, &weight, &input.x)
        })
        .collect()
}

/// The hard gate, before any other check: every slice value finite (an unwritten value is
/// the NaN poison), every byte outside the slice and in the canary still poison, the q8_1
/// pad still poison (the GEMM may read it, never write it). Returns the slice as f64,
/// `[column][row]`.
fn gated_slice(run: &Run, case: &Case) -> Vec<f64> {
    let floats = testkit::floats(&run.dst[..case.m * case.ld * 4]);
    let mut slice = Vec::with_capacity(case.m * case.n);
    for j in 0..case.m {
        for i in 0..case.ld {
            let v = floats[j * case.ld + i];
            if (case.col_off..case.col_off + case.n).contains(&i) {
                assert!(
                    v.is_finite(),
                    "{}: dst[{j}][{i}] is {v}; an unwritten slice value is the NaN poison",
                    case.name
                );
                slice.push(f64::from(v));
            } else {
                assert_eq!(
                    v.to_bits(),
                    0xffff_ffff,
                    "{}: dst[{j}][{i}] outside the slice was written",
                    case.name
                );
            }
        }
    }
    assert!(
        run.dst[case.m * case.ld * 4..].iter().all(|b| *b == POISON),
        "{}: the canary after dst was written",
        case.name
    );
    let blocks = (case.k / 128) * case.m * 144;
    assert!(
        run.y[blocks..].iter().all(|b| *b == POISON),
        "{}: the q8_1 pad was written",
        case.name
    );
    assert!(
        run.fixup_tail.iter().all(|b| *b == POISON),
        "{}: the canary after the fixup range was written",
        case.name
    );
    slice
}

/// The f64 reference `[column][row]`: the dequantized weights (the library's own host
/// dequantizer) against the unquantized f32 activations, rows split over threads.
fn reference(case: &Case, w: &[u8], x: &[f32]) -> Vec<f64> {
    let weights = dequantize(gguf_of(case.ty), w).expect("the host dequantizer");
    let xs: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
    let (n, k, m) = (case.n, case.k, case.m);
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get().min(16));
    let per = n.div_ceil(threads);
    let parts: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (weights, xs) = (&weights, &xs);
                scope.spawn(move || {
                    let (lo, hi) = ((t * per).min(n), ((t + 1) * per).min(n));
                    let mut part = vec![0f64; (hi - lo) * m];
                    for i in lo..hi {
                        let row = &weights[i * k..(i + 1) * k];
                        for j in 0..m {
                            let col = &xs[j * k..(j + 1) * k];
                            part[(i - lo) * m + j] =
                                row.iter().zip(col).map(|(a, b)| f64::from(*a) * b).sum();
                        }
                    }
                    part
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("reference thread"))
            .collect()
    });
    let mut out = vec![0f64; m * n];
    for (t, part) in parts.iter().enumerate() {
        for (r, row) in part.chunks(m).enumerate() {
            for (j, v) in row.iter().enumerate() {
                out[j * n + t * per + r] = *v;
            }
        }
    }
    out
}

/// The energy-weighted NMSE of a whole case and the spread of its per-column values.
struct Nmse {
    global: f64,
    p50: f64,
    p99: f64,
    max: f64,
    /// The share of columns whose own NMSE is above the frozen dense Q8_0 bound.
    share_above: f64,
}

fn nmse(dev: &[f64], want: &[f64], n: usize, m: usize) -> Nmse {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let mut per_column = Vec::with_capacity(m);
    for j in 0..m {
        let (mut cn, mut cd) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (d, r) = (dev[j * n + i], want[j * n + i]);
            cn += (d - r) * (d - r);
            cd += r * r;
        }
        assert!(
            cd > 0.0 && cd.is_finite(),
            "column {j} has no reference energy"
        );
        per_column.push(cn / cd);
        num += cn;
        den += cd;
    }
    per_column.sort_by(f64::total_cmp);
    let above = per_column.iter().filter(|v| **v > Q8X_DENSE_Q80).count();
    Nmse {
        share_above: above as f64 / m as f64,
        global: num / den,
        p50: per_column[m / 2],
        p99: per_column[(m * 99 / 100).min(m - 1)],
        max: per_column[m - 1],
    }
}

fn input(case: Case, seed: u64) -> CaseInput {
    let w = weight_bytes(case.ty, case.n, case.k, seed);
    let x = activations(case.m, case.k, seed ^ 0x5555);
    assert!(
        x.iter().all(|v| v.is_finite()),
        "the activations must be finite"
    );
    CaseInput { case, w, x }
}

fn dense(name: &'static str, ty: QuantType, shape: (usize, usize, usize)) -> Case {
    let (k, n, m) = shape;
    Case {
        name,
        ty,
        k,
        n,
        m,
        ld: n,
        col_off: 0,
    }
}

#[test]
fn mmq_dense_q8x_matches_the_f64_reference_per_class() {
    // Plan section B.5 item 1 at the real Qwen dense shapes (K x N, M columns): the six
    // Q8_0 classes of A.4. Every case is stream-k or tiling by its own shape; the 2048x32
    // class has fallback on (32 % 128), the others off.
    const NAME: &str = "mmq_dense_q8x_matches_the_f64_reference_per_class";
    if !testkit::device_present(NAME) {
        return;
    }
    let classes = [
        ("attn_qkv 2048x8192", (2048, 8192, 64)),
        ("ssm_in 2048x4096", (2048, 4096, 64)),
        ("attn_out 4096x2048", (4096, 2048, 64)),
        ("shared_up 2048x512", (2048, 512, 512)),
        ("shared_down 512x2048", (512, 2048, 512)),
        ("ssm_alpha 2048x32", (2048, 32, 512)),
    ];
    let inputs: Vec<CaseInput> = classes
        .iter()
        .zip(0u64..)
        .map(|(&(name, shape), i)| input(dense(name, QuantType::Q80, shape), 0x6451_0000 + i))
        .collect();
    let runs = run_cases(&inputs).expect("the device phase");
    for (input, run) in inputs.iter().zip(&runs) {
        let (case, dev) = (&input.case, gated_slice(run, &input.case));
        let want = reference(case, &input.w, &input.x);
        let stat = nmse(&dev, &want, case.n, case.m);
        println!(
            "Q8X dense {}: energy-weighted NMSE {:.3e}, per-column p50 {:.3e} p99 {:.3e} max \
             {:.3e}, share of columns above the bound {:.1} % (frozen bound {Q8X_DENSE_Q80:.0e})",
            case.name,
            stat.global,
            stat.p50,
            stat.p99,
            stat.max,
            100.0 * stat.share_above
        );
        assert!(
            stat.global <= Q8X_DENSE_Q80,
            "{}: NMSE {:.3e} exceeds the frozen {Q8X_DENSE_Q80:.0e}",
            case.name,
            stat.global
        );
    }
}

#[test]
fn mmq_dense_k_quant_rows_are_gated_for_finiteness_and_reported() {
    // Q6_K and Q4_K have no frozen dense bound (plan section A.4 item 3: report-only);
    // the hard gate still applies to every value.
    const NAME: &str = "mmq_dense_k_quant_rows_are_gated_for_finiteness_and_reported";
    if !testkit::device_present(NAME) {
        return;
    }
    let inputs = [
        input(
            dense("Q6_K 2048x2048", QuantType::Q6K, (2048, 2048, 64)),
            0x6452_0001,
        ),
        input(
            dense("Q6_K 2048x4096", QuantType::Q6K, (2048, 4096, 130)),
            0x6452_0002,
        ),
        input(
            dense("Q4_K 2048x2048", QuantType::Q4K, (2048, 2048, 64)),
            0x6452_0003,
        ),
        input(
            dense("Q4_K 4096x512", QuantType::Q4K, (4096, 512, 130)),
            0x6452_0004,
        ),
    ];
    let runs = run_cases(&inputs).expect("the device phase");
    for (input, run) in inputs.iter().zip(&runs) {
        let (case, dev) = (&input.case, gated_slice(run, &input.case));
        let want = reference(case, &input.w, &input.x);
        let stat = nmse(&dev, &want, case.n, case.m);
        println!(
            "dense {} (report only): NMSE {:.3e}, per-column p50 {:.3e} p99 {:.3e} max {:.3e}",
            case.name, stat.global, stat.p50, stat.p99, stat.max
        );
    }
}

#[test]
fn mmq_dense_writes_only_its_slice_of_a_fused_destination_deterministically() {
    // A GEMM writes its slice of a fused [m][ld] destination (`dst = base + col_off`,
    // `nrows_dst = ld`): the bytes outside the slice stay poison, the slice meets the
    // frozen Q8X bound, and the same inputs twice give the same bytes (the stream-k
    // decomposition is a function of shape and SM count only; plan section B.5 item 6).
    const NAME: &str = "mmq_dense_writes_only_its_slice_of_a_fused_destination_deterministically";
    if !testkit::device_present(NAME) {
        return;
    }
    let case = Case {
        name: "fused Q8_0 512 of 1024",
        ty: QuantType::Q80,
        k: 2048,
        n: 512,
        m: 100,
        ld: 1024,
        col_off: 256,
    };
    let first = input(case.clone(), 0x6453_0001);
    let again = CaseInput {
        case: case.clone(),
        w: first.w.clone(),
        x: first.x.clone(),
    };
    let k_quant = Case {
        name: "fused Q4_K 256 of 640",
        ty: QuantType::Q4K,
        k: 2048,
        n: 256,
        m: 50,
        ld: 640,
        col_off: 128,
    };
    let inputs = [first, again, input(k_quant, 0x6453_0002)];
    let runs = run_cases(&inputs).expect("the device phase");
    let slices: Vec<Vec<f64>> = inputs
        .iter()
        .zip(&runs)
        .map(|(input, run)| gated_slice(run, &input.case))
        .collect();
    assert_eq!(
        runs[0].dst, runs[1].dst,
        "the same inputs must give the same bytes"
    );
    let want = reference(&inputs[0].case, &inputs[0].w, &inputs[0].x);
    let stat = nmse(&slices[0], &want, case.n, case.m);
    println!(
        "fused Q8_0 slice: NMSE {:.3e} (frozen bound {Q8X_DENSE_Q80:.0e})",
        stat.global
    );
    assert!(
        stat.global <= Q8X_DENSE_Q80,
        "fused slice NMSE {:.3e}",
        stat.global
    );
    let want = reference(&inputs[2].case, &inputs[2].w, &inputs[2].x);
    let stat = nmse(&slices[2], &want, 256, 50);
    println!("fused Q4_K slice (report only): NMSE {:.3e}", stat.global);
}

// ---- row invariance, the replica/launch coupling, the memcheck shapes, speed ---------------

/// The facts of the device the phase ran on, for the host-side replica of the launch.
#[derive(Clone, Copy, Debug)]
struct Facts {
    cc: i32,
    smpbo: usize,
    nsm: usize,
}

/// The launch's tile plan for a dense call, from the G1 replica and the vendored rules
/// (mmq.cuh:1396-1408): the tile width, the tile count, and whether the launch runs the
/// tiling kernel (efficiency >= 90 %) or stream-k with a fixup.
#[derive(Debug, PartialEq, Eq)]
struct Plan {
    j: usize,
    tiles: usize,
    tiling: bool,
    fixup: bool,
}

fn plan_of(case: &Case, facts: Facts) -> Plan {
    let fallback = !case.n.is_multiple_of(128);
    let j = j_best(
        mmq_type_of(case.ty),
        fallback,
        facts.cc,
        facts.smpbo,
        case.m,
    )
    .expect("a tile width fits the shared memory");
    let tiles = case.m.div_ceil(j) * case.n.div_ceil(128);
    let waves = tiles.div_ceil(facts.nsm);
    let tiling = 100 * tiles / (facts.nsm * waves) >= 90;
    Plan {
        j,
        tiles,
        tiling,
        fixup: !tiling && !tiles.is_multiple_of(facts.nsm),
    }
}

/// The device facts of `ctx` through G4's per-context method (read once, then cached); the
/// one place the tests ask for them.
fn facts_of(ctx: &CudaContext) -> Result<Facts, EngError> {
    let facts = ctx.mmq_device_facts()?;
    Ok(Facts {
        cc: facts.cc,
        smpbo: facts.smpbo,
        nsm: facts.nsm,
    })
}

/// A permutation of `0..m` that only moves a column inside its own block of `bounds`
/// (`[0, 80, 130]` is two blocks); `perm[j]` is where column `j` goes.
fn block_permutation(bounds: &[usize], seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut perm: Vec<usize> = (0..*bounds.last().expect("bounds")).collect();
    for block in bounds.windows(2) {
        for i in (block[0] + 1..block[1]).rev() {
            let j = block[0] + rng.below((i - block[0] + 1) as u64) as usize;
            perm.swap(i, j);
        }
    }
    perm
}

/// The activation rows with row `j` moved to row `perm[j]`.
fn permuted_rows(x: &[f32], k: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (j, &to) in perm.iter().enumerate() {
        out[to * k..(to + 1) * k].copy_from_slice(&x[j * k..(j + 1) * k]);
    }
    out
}

/// Columns `j` of `base` whose bytes equal column `perm[j]` of `moved`.
fn equal_columns(base: &Run, moved: &Run, case: &Case, perm: &[usize]) -> usize {
    let column = |run: &Run, j: usize| {
        let at = j * case.ld * 4;
        run.dst[at..at + case.n * 4].to_vec()
    };
    perm.iter()
        .enumerate()
        .filter(|&(j, &to)| column(base, j) == column(moved, to))
        .count()
}

/// The phase of a row-invariance test: the base run, one run per permutation, the facts.
fn permuted_phase(input: &CaseInput, perms: &[Vec<usize>]) -> Result<(Vec<Run>, Facts), EngError> {
    let rig = open_rig()?;
    let case = &input.case;
    let weight = upload_weight(&rig.ctx, case.ty, (case.n, case.k), &input.w)?;
    let mut runs = vec![run_case(&rig, case, &weight, &input.x)?];
    for perm in perms {
        let x = permuted_rows(&input.x, case.k, perm);
        runs.push(run_case(&rig, case, &weight, &x)?);
    }
    Ok((runs, facts_of(&rig.ctx)?))
}

#[test]
fn mmq_dense_rows_are_invariant_within_a_tile_and_across_tiles_in_tiling_mode() {
    // Plan section B.5 item 3 (PG1b), scope of rev 3 (MSG-85 M3): bit-exact per FIXED tile
    // configuration, where the configuration includes the stream-k decomposition. A dense
    // stream-k shape permutes only inside a J-tile; a tiling-mode shape (efficiency >= 90 %,
    // no fixup) permutes across tile boundaries too. Both runs of a pair have the same M, so
    // they are one configuration. The stream-k shape's cross-tile permutation is reported,
    // not graded: the split tile's sum is a different addition order (mmq.cuh:1246, :1290).
    const NAME: &str = "mmq_dense_rows_are_invariant_within_a_tile_and_across_tiles_in_tiling_mode";
    if !testkit::device_present(NAME) {
        return;
    }
    let stream_k = input(
        dense("stream-k 2048x2048", QuantType::Q80, (2048, 2048, 130)),
        0x6454_0001,
    );
    let tiling = input(
        dense("tiling 2048x5248", QuantType::Q80, (2048, 5248, 130)),
        0x6454_0002,
    );
    let within = block_permutation(&[0, 80, 130], 0x6454_0003);
    let across = block_permutation(&[0, 130], 0x6454_0004);
    let (sk_runs, facts) =
        permuted_phase(&stream_k, &[within.clone(), across.clone()]).expect("stream-k phase");
    let (tl_runs, _) =
        permuted_phase(&tiling, std::slice::from_ref(&across)).expect("tiling phase");
    println!("PG1b facts: {facts:?}");

    // The configurations, from the replica: the tile width 80 for M = 130 and the plan flags.
    let sk_plan = plan_of(&stream_k.case, facts);
    let tl_plan = plan_of(&tiling.case, facts);
    println!("PG1b stream-k plan {sk_plan:?}, tiling plan {tl_plan:?}");
    assert_eq!(
        sk_plan.j, 80,
        "M = 130 picks J = 80 (strict <, plan section B.3)"
    );
    assert!(
        sk_plan.fixup && !sk_plan.tiling,
        "the stream-k row must run the fixup"
    );
    assert!(
        tl_plan.tiling && !tl_plan.fixup,
        "the tiling row must run without the fixup"
    );
    assert_eq!(within[..80].iter().filter(|&&to| to >= 80).count(), 0);

    // The hard gate on every run before any comparison.
    for (run, case) in sk_runs
        .iter()
        .map(|r| (r, &stream_k.case))
        .chain(tl_runs.iter().map(|r| (r, &tiling.case)))
    {
        gated_slice(run, case);
    }
    let m = stream_k.case.m;
    let kept = equal_columns(&sk_runs[0], &sk_runs[1], &stream_k.case, &within);
    println!("PG1b stream-k, within-tile permutation: {kept}/{m} columns bit-equal");
    assert_eq!(
        kept, m,
        "a within-tile permutation must keep every column bit-exact"
    );
    let kept = equal_columns(&tl_runs[0], &tl_runs[1], &tiling.case, &across);
    println!("PG1b tiling mode, cross-tile permutation: {kept}/{m} columns bit-equal");
    assert_eq!(
        kept, m,
        "a tiling-mode permutation must keep every column bit-exact"
    );
    let crossed = equal_columns(&sk_runs[0], &sk_runs[2], &stream_k.case, &across);
    println!(
        "PG1b stream-k, cross-tile permutation (report only): {crossed}/{m} columns bit-equal"
    );
}

/// The coupling row's three launches: the replica-sized one, the production one, and the
/// error text of the J_ub-sized one.
struct Coupling {
    facts: Facts,
    exact: Run,
    maximum: Run,
    refused: Result<(), EngError>,
}

fn coupling_phase(
    input: &CaseInput,
    sizes: impl Fn(Facts) -> (usize, usize),
) -> Result<Coupling, EngError> {
    let rig = open_rig()?;
    let (ctx, stream) = (&rig.ctx, &rig.stream);
    let case = &input.case;
    let facts = facts_of(ctx)?;
    let weight = upload_weight(ctx, case.ty, (case.n, case.k), &input.w)?;
    let need = dense_needs(case.ty, case.k, case.n, case.m, case.ld, case.col_off)?;
    let xb = testkit::upload_f32(ctx, &input.x);
    let y = testkit::upload_filled(ctx, POISON, need.y);
    ctx.mmq_quantize(
        stream,
        MmqQuantizeCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
            ty: mmq_type_of(case.ty),
            k: case.k,
            row_stride: case.k,
            ncols: case.m,
            x: xb.range(0, xb.bytes())?,
            gather: None,
            y: y.range(0, need.y)?,
        },
    )?;
    let (exact_bytes, undersized_bytes) = sizes(facts);
    let launch = |fixup_bytes: usize, floor: usize| -> Result<Run, EngError> {
        let dst = testkit::upload_filled(ctx, POISON, need.dst + CANARY);
        let fixup = fixup_buffer(ctx, fixup_bytes);
        ctx.launch_dense(
            stream,
            MmqDenseCall {
                weight: &weight,
                w_seg: None,
                m: case.m,
                q8_1: y.range(0, need.y)?,
                dst: dst.range(0, need.dst)?,
                ld: case.ld,
                col_off: case.col_off,
                fixup: fixup.range(0, fixup_bytes)?,
            },
            floor,
        )?;
        stream.synchronize()?;
        Ok(Run {
            dst: testkit::read_bytes(ctx, &dst, need.dst + CANARY),
            y: testkit::read_bytes(ctx, &y, need.y),
            fixup_tail: fixup_tail(ctx, &fixup, fixup_bytes),
        })
    };
    let exact = launch(exact_bytes, 0)?;
    let maximum = launch(ctx.mmq_fixup_bytes()?, 0)?;
    // The refusal comes last: a native failure latches the backend for the process.
    let refused = launch(undersized_bytes, 0).map(|_| ());
    Ok(Coupling {
        facts,
        exact,
        maximum,
        refused,
    })
}

#[test]
fn mmq_dense_fixup_range_follows_the_tile_width_the_launch_picks() {
    // Plan section B.0 (c) and B.5 item 7: at ncols_opt = 24 with fallback on (N = 32) the
    // replica's tile width is 32 while rev 2's upper bound J_ub is 24. A launch whose fixup
    // range is sized for exactly the replica's J succeeds, and the same launch sized for
    // J_ub is refused before any kernel is enqueued: by the C entry's own fixup rule, an
    // invalid argument (status -1), since lane C's ruling on MSG-85 B2; the vendored require
    // (mmq.cuh:1409, cudaErrorInvalidValue) is the backstop behind it. If the replica and
    // the tree ever disagree, one of the two halves fails (mutant V8 sizes the first half
    // with J_ub).
    const NAME: &str = "mmq_dense_fixup_range_follows_the_tile_width_the_launch_picks";
    if !testkit::device_present(NAME) {
        return;
    }
    // process-isolated: latching_test_in_child. The refused launch latches the backend for the
    // process, so the body runs in a child and the parent re-checks the refusal text.
    let path = format!("{}::{NAME}", module_path!().split_once("::").unwrap().1);
    if !testkit::latching_test_in_child(&path, &["coupling refusal: Backend", "(status -1)"]) {
        return;
    }
    let case = dense("coupling 2048x32 M24", QuantType::Q80, (2048, 32, 24));
    let input = input(case, 0x6455_0001);
    let sizes = |facts: Facts| {
        let plan = plan_of(&input.case, facts);
        let bytes = |j: usize| fixup_floats_exact(facts.nsm, j, 128).expect("exact fixup") * 4;
        (bytes(plan.j), bytes(j_ub(input.case.m)))
    };
    let out = coupling_phase(&input, sizes).expect("the coupling phase");
    let plan = plan_of(&input.case, out.facts);
    println!(
        "coupling plan {plan:?}, J_ub {}, facts {:?}",
        j_ub(24),
        out.facts
    );
    assert_eq!(
        (plan.j, j_ub(24)),
        (32, 24),
        "the gap row: replica J differs from J_ub"
    );
    assert!(plan.fixup, "the row must run the stream-k fixup: {plan:?}");
    gated_slice(&out.exact, &input.case);
    gated_slice(&out.maximum, &input.case);
    assert_eq!(
        out.exact.dst, out.maximum.dst,
        "the exact-size range must give the production bytes"
    );
    let text = format!(
        "{:?}",
        out.refused
            .expect_err("the J_ub-sized range must be refused")
    );
    println!("coupling refusal: {text}");
    assert!(
        text.starts_with("Backend") && text.contains("(status -1)"),
        "{text}"
    );
}

/// The three raw launches of the C-entry fixup row.
struct RawFixup {
    exact: Run,
    maximum: Run,
    refused: Result<(), EngError>,
}

/// Launches the dense GEMM of `input` straight through the FFI child (`mmq_dense_launch`),
/// with a fixup range of `fixup_bytes`: no Rust wrapper, so no fixup floor and no
/// `MmqDenseCall`, only the child's own range checks and the C entry behind them.
fn raw_launch(
    rig: &Rig,
    input: &CaseInput,
    weight: &NativeWeight,
    fixup_bytes: usize,
) -> Result<Run, EngError> {
    let Rig { ctx, stream, .. } = rig;
    let case = &input.case;
    let need = dense_needs(case.ty, case.k, case.n, case.m, case.ld, case.col_off)?;
    let xb = testkit::upload_f32(ctx, &input.x);
    let y = testkit::upload_filled(ctx, POISON, need.y);
    ctx.mmq_quantize(
        stream,
        MmqQuantizeCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
            ty: mmq_type_of(case.ty),
            k: case.k,
            row_stride: case.k,
            ncols: case.m,
            x: xb.range(0, xb.bytes())?,
            gather: None,
            y: y.range(0, need.y)?,
        },
    )?;
    let dst = testkit::upload_filled(ctx, POISON, need.dst + CANARY);
    let fixup = fixup_buffer(ctx, fixup_bytes);
    let raw = MmqDenseRaw {
        w: ctx.raw_range(&weight.buffer().range(0, need.w)?)?,
        y: ctx.raw_range(&y.range(0, need.y)?)?,
        dst: ctx.raw_range(&dst.range(0, need.dst)?)?,
        fixup: ctx.raw_range(&fixup.range(0, fixup_bytes)?)?,
        ints: check_dense_rules(case.ty, case.k, case.n, case.m, case.ld, case.col_off)?,
    };
    ctx.launch_on(stream, move |stream_raw| mmq_dense_launch(stream_raw, raw))?;
    stream.synchronize()?;
    Ok(Run {
        dst: testkit::read_bytes(ctx, &dst, need.dst + CANARY),
        y: testkit::read_bytes(ctx, &y, need.y),
        fixup_tail: fixup_tail(ctx, &fixup, fixup_bytes),
    })
}

fn raw_fixup_phase(input: &CaseInput) -> Result<(Facts, RawFixup), EngError> {
    let rig = open_rig()?;
    let case = &input.case;
    let facts = facts_of(&rig.ctx)?;
    let weight = upload_weight(&rig.ctx, case.ty, (case.n, case.k), &input.w)?;
    let exact_bytes = fixup_floats_exact(facts.nsm, plan_of(case, facts).j, 128)? * 4;
    let exact = raw_launch(&rig, input, &weight, exact_bytes)?;
    let maximum = raw_launch(&rig, input, &weight, rig.fixup_bytes)?;
    // One 16-byte unit short of the launch's need, the largest refusal that is still aligned.
    // It comes last: an invalid argument latches the backend for the process.
    let refused = raw_launch(&rig, input, &weight, exact_bytes - 16).map(|_| ());
    Ok((
        facts,
        RawFixup {
            exact,
            maximum,
            refused,
        },
    ))
}

#[test]
fn mmq_dense_c_entry_refuses_a_fixup_range_short_of_the_launch_need() {
    // Lane C's ruling on MSG-85 B2: the C ABI is a boundary of its own, so the C entry sizes
    // the stream-k fixup range itself, for the nsm it reads, and does not leave the refusal
    // to the vendored require behind it. This row bypasses the Rust wrapper entirely (no
    // floor, no call struct): it goes through the FFI child's `mmq_dense_launch`. At N = 32,
    // M = 24, K = 2048 the launch runs one tile on `nsm` stream-k blocks with J = 32, so its
    // need is `nsm * 32 * 128` floats. That exact size and the device maximum launch and give
    // the same bytes; 16 bytes less is refused as an invalid argument (status -1, operation
    // "cuda adapter"), where the vendored require alone would answer cudaErrorInvalidValue
    // (status 1, operation "cuda launch"), so the status tells the two layers apart.
    const NAME: &str = "mmq_dense_c_entry_refuses_a_fixup_range_short_of_the_launch_need";
    if !testkit::device_present(NAME) {
        return;
    }
    // process-isolated: latching_test_in_child. The refused launch latches the backend for the
    // process, so the body runs in a child and the parent re-checks the refusal text.
    let path = format!("{}::{NAME}", module_path!().split_once("::").unwrap().1);
    let expect = [
        "raw fixup refusal: Backend",
        "\"cuda adapter\"",
        "(status -1)",
    ];
    if !testkit::latching_test_in_child(&path, &expect) {
        return;
    }
    let input = input(
        dense("raw fixup 2048x32 M24", QuantType::Q80, (2048, 32, 24)),
        0x6458_0001,
    );
    let (facts, out) = raw_fixup_phase(&input).expect("the raw launch phase");
    let plan = plan_of(&input.case, facts);
    println!("raw fixup plan {plan:?}, facts {facts:?}");
    assert!(
        plan.fixup && plan.j == 32,
        "the row must run the stream-k fixup at J 32: {plan:?}"
    );
    gated_slice(&out.exact, &input.case);
    gated_slice(&out.maximum, &input.case);
    assert_eq!(
        out.exact.dst, out.maximum.dst,
        "the exact-size range must give the production bytes"
    );
    let text = format!(
        "{:?}",
        out.refused
            .expect_err("a range short of the need must be refused")
    );
    println!("raw fixup refusal: {text}");
    assert!(
        text.starts_with("Backend")
            && text.contains("\"cuda adapter\"")
            && text.contains("(status -1)"),
        "{text}"
    );
}

/// The two input paths' bytes for one weight, and the refusal text for a Q5_0 weight.
struct InputPaths {
    via_input: Vec<Vec<u8>>,
    direct: Vec<Vec<u8>>,
    q50: Result<(), EngError>,
}

#[test]
fn mmq_dense_input_quantizes_in_the_layout_of_the_weight() {
    // The family-facing input path derives the q8_1 scale layout (D4 for Q8_0 and Q6_K, DS4
    // for Q4_K) from the consuming weight: its bytes equal a direct G4 quantize with that
    // weight's type, the layouts share their codes and differ in their scale words, Q6_K
    // gives Q8_0's bytes, and a Q5_0 weight, which has no dense case, is refused by the dense
    // tag.
    const NAME: &str = "mmq_dense_input_quantizes_in_the_layout_of_the_weight";
    if !testkit::device_present(NAME) {
        return;
    }
    let (k, n, m) = (256usize, 128usize, 8usize);
    let x = activations(m, k, 0x6459_0001);
    let phase = || -> Result<InputPaths, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let need = dense_needs(QuantType::Q80, k, n, m, n, 0)?;
        let xb = testkit::upload_f32(&ctx, &x);
        let mut paths = InputPaths {
            via_input: Vec::new(),
            direct: Vec::new(),
            q50: Ok(()),
        };
        for (ty, mmq) in [
            (QuantType::Q80, MmqType::Q80),
            (QuantType::Q4K, MmqType::Q4K),
            (QuantType::Q6K, MmqType::Q6K),
        ] {
            let weight = upload_weight(&ctx, ty, (n, k), &weight_bytes(ty, n, k, 0x6459_0002))?;
            let (via, direct) = (
                testkit::upload_filled(&ctx, POISON, need.y),
                testkit::upload_filled(&ctx, POISON, need.y),
            );
            ctx.mmq_dense_input(
                &stream,
                MmqDenseInputCall {
                    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                    weight: &weight,
                    m,
                    row_stride: k,
                    x: xb.range(0, xb.bytes())?,
                    y: via.range(0, need.y)?,
                },
            )?;
            ctx.mmq_quantize(
                &stream,
                MmqQuantizeCall {
                    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                    ty: mmq,
                    k,
                    row_stride: k,
                    ncols: m,
                    x: xb.range(0, xb.bytes())?,
                    gather: None,
                    y: direct.range(0, need.y)?,
                },
            )?;
            stream.synchronize()?;
            paths
                .via_input
                .push(testkit::read_bytes(&ctx, &via, need.y));
            paths
                .direct
                .push(testkit::read_bytes(&ctx, &direct, need.y));
        }
        let q50 = upload_weight(&ctx, QuantType::Q50, (n, k), &vec![0u8; n * (k / 32) * 22])?;
        let y = testkit::upload_filled(&ctx, POISON, need.y);
        paths.q50 = ctx.mmq_dense_input(
            &stream,
            MmqDenseInputCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                weight: &q50,
                m,
                row_stride: k,
                x: xb.range(0, xb.bytes())?,
                y: y.range(0, need.y)?,
            },
        );
        Ok(paths)
    };
    let out = phase().expect("the input-path phase");
    let blocks = (k / 128) * m * 144;
    for (via, direct) in out.via_input.iter().zip(&out.direct) {
        // The hard gate: the written blocks are no longer the poison, and the pad still is.
        assert!(
            via[..blocks].iter().any(|b| *b != POISON),
            "the input path wrote nothing"
        );
        assert!(
            via[blocks..].iter().all(|b| *b == POISON),
            "the pad was written"
        );
        assert_eq!(
            via, direct,
            "the input path must equal the direct quantize of its type"
        );
    }
    // The layouts share the codes (qs bytes 16..144 of each block) and differ in the scales.
    let codes = |bytes: &Vec<u8>| -> Vec<u8> {
        bytes[..blocks]
            .chunks(144)
            .flat_map(|block| block[16..].to_vec())
            .collect()
    };
    assert_eq!(codes(&out.via_input[0]), codes(&out.via_input[1]));
    assert_ne!(
        out.via_input[0][..blocks],
        out.via_input[1][..blocks],
        "D4 and DS4 scale words differ"
    );
    // Q6_K takes the D4 layout like Q8_0 (mmq.cuh:61-101): the same bytes, so a Q6_K arm
    // that hands out the DS4 layout of Q4_K fails here and in the direct comparison above.
    assert_eq!(
        out.via_input[2], out.via_input[0],
        "Q6_K and Q8_0 share the D4 layout"
    );
    let text = format!(
        "{:?}",
        out.q50.expect_err("a Q5_0 weight has no dense input")
    );
    assert!(text.contains("mmq_dense: type"), "{text}");
}

#[test]
fn mmq_dense_memcheck_shapes_run_inside_their_buffers() {
    // Plan section B.5 item 4 for the dense entry: ncols_opt in {1, 5, 7, 9, 13, 14} at
    // fallback off, {17, 18, 19} and 24 at fallback on, 130 (J = 80), plus one stream-k
    // fixup shape. The q8_1 range is an exact-size allocation, so `compute-sanitizer
    // --tool memcheck` sees every read past `q8_1_bytes`; the gate checks the canaries.
    const NAME: &str = "mmq_dense_memcheck_shapes_run_inside_their_buffers";
    if !testkit::device_present(NAME) {
        return;
    }
    let mut shapes: Vec<(usize, usize, usize)> = Vec::new();
    shapes.extend([1usize, 5, 7, 9, 13, 14].map(|m| (256, 256, m)));
    shapes.extend([17usize, 18, 19, 24].map(|m| (256, 160, m)));
    shapes.extend([(256, 256, 130), (2048, 384, 24)]);
    let inputs: Vec<CaseInput> = [QuantType::Q80, QuantType::Q4K, QuantType::Q6K]
        .into_iter()
        .zip(0u64..)
        .flat_map(|(ty, t)| {
            shapes.iter().zip(0u64..).map(move |(&(k, n, m), s)| {
                input(
                    dense("memcheck shape", ty, (k, n, m)),
                    0x6456_0000 + 64 * t + s,
                )
            })
        })
        .collect();
    let runs = run_cases(&inputs).expect("the device phase");
    let mut worst_q80 = 0.0f64;
    for (input, run) in inputs.iter().zip(&runs) {
        let (case, dev) = (&input.case, gated_slice(run, &input.case));
        let stat = nmse(&dev, &reference(case, &input.w, &input.x), case.n, case.m);
        if case.ty == QuantType::Q80 {
            worst_q80 = worst_q80.max(stat.global);
        }
    }
    println!(
        "memcheck shapes: {} runs, worst Q8_0 NMSE {worst_q80:.3e}",
        runs.len()
    );
    assert!(
        worst_q80 <= Q8X_DENSE_Q80,
        "memcheck-shape Q8_0 NMSE {worst_q80:.3e}"
    );
}

// ---- speed rows (reported, never gated) -----------------------------------------------------

/// The dense Q8_0 shapes of one 40-layer pass with their counts (CPF-M1a section 3: 30
/// DeltaNet and 10 attention layers, the shared expert in all 40): `(name, k, n, count)`.
const SPEED_SHAPES: [(&str, usize, usize, usize); 9] = [
    ("attn_qkv", 2048, 8192, 30),
    ("attn_gate", 2048, 4096, 30),
    ("ssm_out", 4096, 2048, 30),
    ("ssm_alpha_beta", 2048, 32, 60),
    ("attn_q", 2048, 8192, 10),
    ("attn_k_v", 2048, 512, 20),
    ("attn_output", 4096, 2048, 10),
    ("shexp_gate_up", 2048, 512, 80),
    ("shexp_down", 512, 2048, 40),
];
/// Launches per timed loop, after the warm-up.
const SPEED_REPS: usize = 20;
/// The shape whose output gets an NMSE spot-check in the speed process (verdict n-9; the
/// reference of the others would cost minutes): the same entry and binary the Q8X rows grade.
const SPOT_SHAPE: &str = "ssm_alpha_beta";
/// The seed of the speed rows' activations.
const SPEED_SEED: u64 = 0x6457_0001;

/// Microseconds per launch of one shape at one M: the GEMM alone, and quantize + GEMM.
struct Timing {
    name: &'static str,
    m: usize,
    count: usize,
    gemm_us: f64,
    both_us: f64,
    finite: bool,
    /// The destination of the shape that gets an NMSE spot-check (the small alpha/beta one).
    spot: Option<Vec<f32>>,
}

fn timing_phase(ms: &[usize]) -> Result<Vec<Timing>, EngError> {
    let rig = open_rig()?;
    let Rig {
        ctx,
        stream,
        fixup,
        fixup_bytes,
    } = &rig;
    let mut out = Vec::new();
    for &(name, k, n, count) in &SPEED_SHAPES {
        let bytes = testkit::weight_source(GgufType::Q80, n, k);
        let weight = upload_weight(ctx, QuantType::Q80, (n, k), &bytes)?;
        for &m in ms {
            let need = dense_needs(QuantType::Q80, k, n, m, n, 0)?;
            let xb = testkit::upload_f32(ctx, &activations(m, k, SPEED_SEED));
            let y = testkit::upload_filled(ctx, POISON, need.y);
            let dst = testkit::upload_filled(ctx, POISON, need.dst);
            let quantize = || {
                ctx.mmq_quantize(
                    stream,
                    MmqQuantizeCall {
                        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                        ty: MmqType::Q80,
                        k,
                        row_stride: k,
                        ncols: m,
                        x: xb.range(0, xb.bytes())?,
                        gather: None,
                        y: y.range(0, need.y)?,
                    },
                )
            };
            let gemm = || -> Result<(), EngError> {
                // The launch with the floor already known: the facts query of `mmq_dense`
                // is a once-per-pass host cost, outside the kernel time measured here.
                ctx.launch_dense(
                    stream,
                    MmqDenseCall {
                        weight: &weight,
                        w_seg: None,
                        m,
                        q8_1: y.range(0, need.y)?,
                        dst: dst.range(0, need.dst)?,
                        ld: n,
                        col_off: 0,
                        fixup: fixup.range(0, *fixup_bytes)?,
                    },
                    *fixup_bytes,
                )
            };
            quantize()?;
            let time = |both: bool| -> Result<f64, EngError> {
                for _ in 0..3 {
                    gemm()?;
                }
                stream.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..SPEED_REPS {
                    if both {
                        quantize()?;
                    }
                    gemm()?;
                }
                stream.synchronize()?;
                Ok(start.elapsed().as_secs_f64() * 1e6 / SPEED_REPS as f64)
            };
            let (gemm_us, both_us) = (time(false)?, time(true)?);
            let floats = testkit::floats(&testkit::read_bytes(ctx, &dst, need.dst));
            let finite = floats.iter().all(|v| v.is_finite());
            let spot = (name == SPOT_SHAPE).then_some(floats);
            out.push(Timing {
                name,
                m,
                count,
                gemm_us,
                both_us,
                finite,
                spot,
            });
        }
    }
    Ok(out)
}

#[test]
fn mmq_dense_speed_rows_report() {
    // Plan section B.7 speed rows, reported not gated and UNPAIRED: the dense Q8_0 GEMMs of
    // one 40-layer pass at M = 512 / 1,024 / 2,048, printed beside CPF-M1a's constants
    // 26.39 / 41.78 / 72.73 ms. Those are another day's median of synced single launches on
    // a pinned ggml build; these are the mean of 20 back-to-back launches in this hold, so
    // neither "close to" nor "slower than" stock follows. The paired row comes with the
    // CPF-7c probe. Meaningful in the release profile only: the debug profile compiles the
    // kernels with -G. The hard gate applies to every run, and the alpha/beta output of
    // each M gets an NMSE spot-check against the frozen bound.
    const NAME: &str = "mmq_dense_speed_rows_report";
    if !testkit::device_present(NAME) {
        return;
    }
    let ms = [512usize, 1024, 2048];
    let rows = timing_phase(&ms).expect("the timing phase");
    for row in &rows {
        assert!(
            row.finite,
            "{} at M {}: the destination holds a non-finite value",
            row.name, row.m
        );
        let tops = |us: f64, n: usize, k: usize| 2.0 * (row.m * n * k) as f64 / us / 1e6;
        let (n, k) = SPEED_SHAPES
            .iter()
            .find(|s| s.0 == row.name)
            .map_or((0, 0), |s| (s.2, s.1));
        println!(
            "speed dense {} M={} (unpaired, reported only): gemm {:.1} us ({:.1} TOP/s), \
             quantize+gemm {:.1} us",
            row.name,
            row.m,
            row.gemm_us,
            tops(row.gemm_us, n, k),
            row.both_us
        );
    }
    for (m, anchor) in ms.iter().zip([26.39, 41.78, 72.73]) {
        let sum = |pick: fn(&Timing) -> f64| -> f64 {
            rows.iter()
                .filter(|r| r.m == *m)
                .map(|r| pick(r) * r.count as f64)
                .sum::<f64>()
                / 1e3
        };
        println!(
            "DENSE_TOTAL M={m} (unpaired with the CPF-M1a constant, reported only): gemm only \
             {:.2} ms, quantize+gemm {:.2} ms (CPF-M1a stock {anchor} ms; llama.cpp 1K dense \
             bucket 42.5 ms, design section 2)",
            sum(|r| r.gemm_us),
            sum(|r| r.both_us)
        );
    }
    for row in rows.iter().filter(|r| r.spot.is_some()) {
        let (_, k, n, _) = *SPEED_SHAPES
            .iter()
            .find(|shape| shape.0 == SPOT_SHAPE)
            .expect("the spot-check shape is in the table");
        let case = dense("speed spot", QuantType::Q80, (k, n, row.m));
        let w = testkit::weight_source(GgufType::Q80, n, k);
        let x = activations(row.m, k, SPEED_SEED);
        let dev: Vec<f64> = row.spot.iter().flatten().map(|v| f64::from(*v)).collect();
        let stat = nmse(&dev, &reference(&case, &w, &x), n, row.m);
        println!(
            "speed spot-check M={}: NMSE {:.3e} (frozen bound {Q8X_DENSE_Q80:.0e})",
            row.m, stat.global
        );
        assert!(
            stat.global <= Q8X_DENSE_Q80,
            "M {}: spot NMSE {:.3e}",
            row.m,
            stat.global
        );
    }
}

/// The seam identity the GDN prefill fusion rests on (2026-10-27 divergence): four
/// SEGMENTED launches over one wide upload must be bitwise the four separate-weight
/// launches, because the vendored MMQ schedules stream-k from (n, m, k) - only
/// same-shape launches share a schedule. Also records (no assert: a coinciding schedule
/// is legal) whether the ONE wide launch differs from the four.
#[test]
fn mmq_dense_segments_over_one_upload_match_separate_weights_bitwise() {
    const NAME: &str = "mmq_dense_segments_over_one_upload_match_separate_weights_bitwise";
    if !testkit::device_present(NAME) {
        return;
    }
    const K: usize = 2048;
    const M: usize = 512;
    const SEGS: [(usize, usize); 4] = [(8192, 0), (4096, 8192), (32, 12288), (32, 12320)];
    const LD: usize = 12352;
    struct Out {
        separate: Vec<u8>,
        segmented: Vec<u8>,
        wide: Vec<u8>,
    }
    let phase = || -> Result<Out, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        println!("CUDA devices visible: 1");
        // The same four weight segments in both arms; the wide upload is their concat.
        let segments: Vec<Vec<u8>> = SEGS
            .iter()
            .enumerate()
            .map(|(i, &(rows, _))| q8_0_weight(rows, K, 0xABCD_0000 + i as u64))
            .collect();
        let mut concat = Vec::new();
        for segment in &segments {
            concat.extend_from_slice(segment);
        }
        let wide = upload_weight(&ctx, QuantType::Q80, (LD, K), &concat)?;
        let split: Vec<NativeWeight> = segments
            .iter()
            .map(|bytes| {
                upload_weight(
                    &ctx,
                    QuantType::Q80,
                    (bytes.len() / (K / 32 * 34), K),
                    bytes,
                )
            })
            .collect::<Result<_, _>>()?;
        let host_rows = activations(M, K, 0x600D_0001);
        let x = ctx.upload(
            &host_rows
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let need = dense_needs(QuantType::Q80, K, LD, M, LD, 0)?;
        // Sized from the call's own q8_1 need, not the fixup floor (few-SM devices would
        // under-allocate and refuse before testing anything).
        let y = ctx.allocate(need.y)?;
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q80,
                k: K,
                row_stride: K,
                ncols: M,
                x: x.range(0, M * K * 4)?,
                gather: None,
                y: y.range(0, need.y)?,
            },
        )?;
        let fixup = ctx.allocate(ctx.mmq_fixup_bytes()?)?;
        let mut separate = Vec::new();
        let mut segmented = Vec::new();
        let mut wide_bytes = Vec::new();
        for (arm, into) in [("separate", &mut separate), ("segmented", &mut segmented)] {
            let dst = ctx.allocate(M * LD * 4)?;
            for (i, &(rows, col_off)) in SEGS.iter().enumerate() {
                let (weight, w_seg) = if arm == "separate" {
                    (&split[i], None)
                } else {
                    (&wide, Some((col_off, rows)))
                };
                ctx.mmq_dense(
                    &stream,
                    MmqDenseCall {
                        weight,
                        w_seg,
                        m: M,
                        q8_1: y.range(0, need.y)?,
                        dst: dst.range(0, (M - 1) * LD * 4 + (col_off + rows) * 4)?,
                        ld: LD,
                        col_off,
                        fixup: fixup.range(0, fixup.bytes())?,
                    },
                )?;
            }
            let mut bytes = vec![0u8; dst.bytes()];
            dst.read_at(&stream, 0, &mut bytes)?;
            into.extend_from_slice(&bytes);
        }
        {
            let dst = ctx.allocate(M * LD * 4)?;
            ctx.mmq_dense(
                &stream,
                MmqDenseCall {
                    weight: &wide,
                    w_seg: None,
                    m: M,
                    q8_1: y.range(0, need.y)?,
                    dst: dst.range(0, need.dst)?,
                    ld: LD,
                    col_off: 0,
                    fixup: fixup.range(0, fixup.bytes())?,
                },
            )?;
            let mut bytes = vec![0u8; dst.bytes()];
            dst.read_at(&stream, 0, &mut bytes)?;
            wide_bytes.extend_from_slice(&bytes);
        }
        Ok(Out {
            separate,
            segmented,
            wide: wide_bytes,
        })
    };
    // The solo matrix: each segment alone on a fresh dst, separate vs segmented, so a
    // clobbered neighbor cannot mask which (w_off, n) combinations are themselves broken.
    // Reports differing columns inside the segment's own range and outside it (clobber).
    {
        let ctx = CudaContext::new().expect("the solo context");
        let stream = ctx.stream().expect("the solo stream");
        let segments: Vec<Vec<u8>> = SEGS
            .iter()
            .enumerate()
            .map(|(i, &(rows, _))| q8_0_weight(rows, K, 0xABCD_0000 + i as u64))
            .collect();
        let mut concat = Vec::new();
        for segment in &segments {
            concat.extend_from_slice(segment);
        }
        let wide = upload_weight(&ctx, QuantType::Q80, (LD, K), &concat).expect("wide");
        let split: Vec<NativeWeight> = segments
            .iter()
            .map(|bytes| {
                upload_weight(
                    &ctx,
                    QuantType::Q80,
                    (bytes.len() / (K / 32 * 34), K),
                    bytes,
                )
            })
            .collect::<Result<_, _>>()
            .expect("split");
        let host_rows = activations(M, K, 0x600D_0001);
        let x = ctx
            .upload(
                &host_rows
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("x");
        let need = dense_needs(QuantType::Q80, K, LD, M, LD, 0).expect("needs");
        let y = ctx.allocate(need.y).expect("y");
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q80,
                k: K,
                row_stride: K,
                ncols: M,
                x: x.range(0, M * K * 4).expect("x range"),
                gather: None,
                y: y.range(0, need.y).expect("y range"),
            },
        )
        .expect("quantize");
        let fixup = ctx
            .allocate(ctx.mmq_fixup_bytes().expect("fixup bytes"))
            .expect("fixup");
        let launch = |weight: &NativeWeight,
                      w_seg: Option<(usize, usize)>,
                      rows: usize,
                      col_off: usize|
         -> Vec<u8> {
            let dst = ctx.allocate(M * LD * 4).expect("dst");
            ctx.mmq_dense(
                &stream,
                MmqDenseCall {
                    weight,
                    w_seg,
                    m: M,
                    q8_1: y.range(0, need.y).expect("y range"),
                    dst: dst
                        .range(0, (M - 1) * LD * 4 + (col_off + rows).max(64) * 4)
                        .expect("d"),
                    ld: LD,
                    col_off,
                    fixup: fixup.range(0, fixup.bytes()).expect("f"),
                },
            )
            .expect("the solo launch");
            let mut bytes = vec![0u8; dst.bytes()];
            dst.read_at(&stream, 0, &mut bytes).expect("read");
            bytes
        };
        for (i, &(rows, col_off)) in SEGS.iter().enumerate() {
            let need_solo = dense_needs(QuantType::Q80, K, rows, M, LD, col_off).expect("n");
            println!(
                "{NAME} solo[{i}] n{rows}: need.w {} (rows {}/{}) sepbuf {} widebuf {}",
                need_solo.w,
                need_solo.w / (K / 32 * 34),
                rows,
                split[i].buffer().bytes(),
                wide.buffer().bytes()
            );
            let sep = launch(&split[i], None, rows, col_off);
            let seg = launch(&wide, Some((col_off, rows)), rows, col_off);
            let mut inside = 0usize;
            let mut outside = 0usize;
            let mut first_outside = None;
            for row in 0..M {
                for col in 0..LD {
                    let at = (row * LD + col) * 4;
                    if sep[at..at + 4] != seg[at..at + 4] {
                        if (col_off..col_off + rows).contains(&col) {
                            inside += 1;
                        } else {
                            outside += 1;
                            first_outside.get_or_insert(col);
                        }
                    }
                }
            }
            println!(
                "{NAME} solo[{i}] n{rows} off{col_off}: inside {inside}, outside {outside} (first col {:?})",
                first_outside
            );
            if let Some((row, col)) = (0..M)
                .flat_map(|row| (col_off..col_off + rows).map(move |col| (row, col)))
                .find(|&(row, col)| {
                    let at = (row * LD + col) * 4;
                    sep[at..at + 4] != seg[at..at + 4]
                })
            {
                let at = (row * LD + col) * 4;
                let a = f32::from_le_bytes(sep[at..at + 4].try_into().unwrap());
                let b = f32::from_le_bytes(seg[at..at + 4].try_into().unwrap());
                println!(
                    "{NAME} solo[{i}] first diff row {row} col {col}: sep {a:e} ({:#010x}) seg {b:e} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    let out = phase().expect("the device phase");
    let mut report = vec![format!("{NAME}:")];
    for &(rows, col_off) in SEGS.iter() {
        let mut differing = 0usize;
        let mut max_ulp = 0u32;
        let mut columns = std::collections::BTreeSet::new();
        for row in 0..M {
            for col in col_off..col_off + rows {
                let at = (row * LD + col) * 4;
                let a = f32::from_le_bytes(out.separate[at..at + 4].try_into().unwrap());
                let b = f32::from_le_bytes(out.segmented[at..at + 4].try_into().unwrap());
                if a.to_bits() != b.to_bits() {
                    differing += 1;
                    columns.insert(col);
                    max_ulp = max_ulp.max(a.to_bits().abs_diff(b.to_bits()));
                }
            }
        }
        let span = columns
            .iter()
            .next()
            .zip(columns.iter().next_back())
            .map(|(first, last)| format!(" cols {first}..={last}"))
            .unwrap_or_default();
        report.push(format!(
            "seg[col {col_off}, n {rows}]: {differing} floats differ, {} columns{span}, max ulp {max_ulp}",
            columns.len()
        ));
    }
    let report = report.join(" ");
    println!("{report}");
    assert!(
        out.separate == out.segmented,
        "{NAME}: segmented != separate ({report})"
    );
    let differing = out
        .separate
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.wide.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "{NAME}: segmented==separate bitwise; the ONE wide launch differs in {differing} of {} floats",
        out.separate.len() / 4
    );
}
