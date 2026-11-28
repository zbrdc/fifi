//! The decode dq matvec against the CPU dequant reference of `resident/matvec_q/tests.rs`: every
//! weight value decoded on the host in f64, times the same f32 row, within `64 * 2^-24 * sum |w x|`
//! (the tolerance style of the int8 matvec tests).

use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{CudaContext, DeviceStream};
use crate::kernels::cuda::resident::matvec_dq::MatvecDqCall;
use crate::kernels::cuda::resident::matvec_q::MatvecCall;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::resident::matvec_q::tests::{assert_bad_request, reference};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, testkit};

const TYPES: [QuantType; 3] = [QuantType::Q4K, QuantType::Q6K, QuantType::Q80];
/// k on the kernel's grid, from one K block to the largest decode shape.
const KS: [usize; 5] = [256, 512, 2048, 4096, 12288];
/// Rows: one warp, one warp and a lane group, two blocks and a partly served third, 62 blocks
/// and a partly served 63rd.
const ROWS: [usize; 4] = [4, 5, 37, 1000];
const F32: usize = size_of::<f32>();

/// A weight of `rows` rows in native blocks and one f32 row of `k` values, on the device, with
/// the host bytes of both.
struct Rig {
    ty: QuantType,
    k: usize,
    rows: usize,
    w_bytes: Vec<u8>,
    x: Vec<f32>,
    w: DeviceBuffer,
    x_dev: DeviceBuffer,
}

fn rig(ctx: &CudaContext, ty: QuantType, k: usize, rows: usize, seed: u64) -> Rig {
    let mut rng = testkit::Rng::new(seed);
    let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
    let x: Vec<f32> = (0..k).map(|_| rng.normal() as f32).collect();
    let w = testkit::upload_bytes(ctx, &w_bytes);
    let x_dev = testkit::upload_f32(ctx, &x);
    Rig {
        ty,
        k,
        rows,
        w_bytes,
        x,
        w,
        x_dev,
    }
}

fn call<'b>(
    rig: &Rig,
    w: DeviceRange<'b>,
    x: DeviceRange<'b>,
    y: DeviceRange<'b>,
) -> MatvecDqCall<'b> {
    MatvecDqCall {
        ty: rig.ty,
        w,
        x,
        y,
        k: rig.k,
        rows: rig.rows,
    }
}

/// One launch over the rig into a fresh zeroed `y`; the stream drains and `y` comes back.
fn launch(ctx: &CudaContext, stream: &DeviceStream, rig: &Rig) -> Vec<f32> {
    let w_bytes = rig.ty.weight_bytes(rig.rows, rig.k).expect("size");
    let y_bytes = rig.rows * F32;
    let y = testkit::upload_filled(ctx, 0, y_bytes);
    ctx.matvec_dq(
        stream,
        call(
            rig,
            rig.w.range(0, w_bytes).expect("w range"),
            rig.x_dev.range(0, rig.k * F32).expect("x range"),
            y.range(0, y_bytes).expect("y range"),
        ),
    )
    .expect("matvec_dq");
    stream.synchronize().expect("drain");
    testkit::floats(&testkit::read_bytes(ctx, &y, y_bytes))
}

/// Every row of `got` against the f64 dot of the rig's decoded weights and its x, within
/// `64 * 2^-24 * sum |w x|`; `max_ratio` keeps the worst error over tolerance.
fn check_rows(rig: &Rig, got: &[f32], max_ratio: &mut f64) {
    let (ty, k, rows) = (rig.ty, rig.k, rig.rows);
    let row_bytes = ty.weight_bytes(1, k).expect("row bytes");
    let block_size = ty.block_size();
    for (row, got) in got.iter().enumerate() {
        let w_row = &rig.w_bytes[row * row_bytes..][..row_bytes];
        let (mut want, mut abs) = (0f64, 0f64);
        for (block, x) in w_row
            .chunks_exact(ty.block_bytes())
            .zip(rig.x.chunks_exact(block_size))
        {
            for (i, x) in x.iter().enumerate() {
                let term = reference::weight_value(ty, block, i) * f64::from(*x);
                want += term;
                abs += term.abs();
            }
        }
        let tolerance = 64.0 * 2f64.powi(-24) * abs;
        let error = (f64::from(*got) - want).abs();
        *max_ratio = (*max_ratio).max(error / tolerance);
        assert!(
            error <= tolerance,
            "{ty:?} k {k} rows {rows} row {row}: error {error} over tol {tolerance}"
        );
    }
}

/// Both types over the k and row lists, against the CPU reference.
#[test]
fn matvec_dq_matches_the_cpu_dequantized_reference() {
    if !testkit::device_present("matvec_dq_matches_the_cpu_dequantized_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut max_ratio = 0f64;
    for (ti, ty) in TYPES.into_iter().enumerate() {
        for (ki, k) in KS.into_iter().enumerate() {
            for (ri, rows) in ROWS.into_iter().enumerate() {
                let rig = rig(
                    &ctx,
                    ty,
                    k,
                    rows,
                    (0xD0_0000 + 100 * ti + 10 * ki + ri) as u64,
                );
                check_rows(&rig, &launch(&ctx, &stream, &rig), &mut max_ratio);
            }
        }
    }
    println!("matvec dq: max tolerance ratio {max_ratio:.3}");
}

/// 37 rows leave the third block's warps wholly or partly without rows: a canary of 64 bytes
/// after y must stay as it was, and the 37 rows must be written.
#[test]
fn matvec_dq_writes_only_its_rows() {
    if !testkit::device_present("matvec_dq_writes_only_its_rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (ti, ty) in TYPES.into_iter().enumerate() {
        let rig = rig(&ctx, ty, 2048, 37, 0xD1_0000 + ti as u64);
        let (y_bytes, guard) = (37 * F32, 64);
        let y = testkit::upload_filled(&ctx, 0xAA, y_bytes + guard);
        let w_bytes = ty.weight_bytes(37, 2048).expect("size");
        ctx.matvec_dq(
            &stream,
            call(
                &rig,
                rig.w.range(0, w_bytes).expect("w range"),
                rig.x_dev.range(0, 2048 * F32).expect("x range"),
                y.range(0, y_bytes).expect("y range"),
            ),
        )
        .expect("matvec_dq");
        stream.synchronize().expect("drain");
        let bytes = testkit::read_bytes(&ctx, &y, y_bytes + guard);
        for (i, byte) in bytes[y_bytes..].iter().enumerate() {
            assert_eq!(*byte, 0xAA, "{ty:?}: canary {i} beyond y was written");
        }
        for (row, got) in testkit::floats(&bytes[..y_bytes]).iter().enumerate() {
            assert_ne!(
                got.to_bits(),
                0xAAAA_AAAA,
                "{ty:?}: row {row} was not written"
            );
        }
    }
}

/// Two launches of the same weight and row, each into its own allocation, give the same bits.
#[test]
fn matvec_dq_gives_the_same_bits_on_every_launch() {
    if !testkit::device_present("matvec_dq_gives_the_same_bits_on_every_launch") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (ti, ty) in TYPES.into_iter().enumerate() {
        let rig = rig(&ctx, ty, 4096, 1000, 0xD2_0000 + ti as u64);
        let first = launch(&ctx, &stream, &rig);
        let second = launch(&ctx, &stream, &rig);
        for row in 0..rig.rows {
            assert_eq!(
                first[row].to_bits(),
                second[row].to_bits(),
                "{ty:?}: the second launch differs at row {row}"
            );
        }
    }
}

/// The wrapper's refusals, each a BadRequest before any native call: the shape, the three
/// ranges' sizes and alignments, and y sharing a byte with w or x. The native entry's own
/// refusals (null pointers included) are pinned in `native/resident/matvec_dq.rs`.
#[test]
fn matvec_dq_refuses_bad_arguments_before_the_adapter() {
    if !testkit::device_present("matvec_dq_refuses_bad_arguments_before_the_adapter") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rig = rig(&ctx, QuantType::Q4K, 2048, 64, 0xD3_0001);
    let (x_bytes, y_bytes) = (2048 * F32, 64 * F32);
    let w_bytes = QuantType::Q4K.weight_bytes(64, 2048).expect("size");
    let wide_w = testkit::upload_filled(&ctx, 0, w_bytes + 16);
    let skewed_x = testkit::upload_filled(&ctx, 0, x_bytes + 16);
    let y = testkit::upload_filled(&ctx, 0, y_bytes + 4);
    let (w, x, y_ok) = (
        rig.w.range(0, w_bytes).expect("w range"),
        rig.x_dev.range(0, x_bytes).expect("x range"),
        y.range(0, y_bytes).expect("y range"),
    );
    ctx.matvec_dq(&stream, call(&rig, w, x, y_ok))
        .expect("the base call launches");
    stream.synchronize().expect("drain");
    let refused = |call: MatvecDqCall<'_>, case: &str, needle: &str| {
        assert_bad_request(ctx.matvec_dq(&stream, call), case, needle)
    };
    let shape = "k a positive multiple of 256";
    let mut c = call(&rig, w, x, y_ok);
    c.k = 100;
    refused(c, "k not a multiple of 256", shape);
    let mut c = call(&rig, w, x, y_ok);
    c.rows = 0;
    refused(c, "rows 0", shape);
    let mut c = call(&rig, w, x, y_ok);
    c.ty = QuantType::Q5K;
    refused(c, "a type the kernel does not serve", shape);

    let mut c = call(&rig, w, x, y_ok);
    c.x = rig.x_dev.range(0, x_bytes - 4).expect("x range");
    refused(
        c,
        "x one float short",
        "matvec_dq x range holds 8188 bytes but the call needs 8192",
    );
    let mut c = call(&rig, w, x, y_ok);
    c.w = rig.w.range(0, w_bytes - 144).expect("w range");
    refused(
        c,
        "w one block short",
        &format!(
            "matvec_dq w range holds {} bytes but the call needs {w_bytes}",
            w_bytes - 144
        ),
    );
    let mut c = call(&rig, w, x, y_ok);
    c.y = y.range(0, y_bytes - 4).expect("y range");
    refused(
        c,
        "y one float short",
        "matvec_dq y range holds 252 bytes but the call needs 256",
    );

    let mut c = call(&rig, w, x, y_ok);
    c.w = wide_w.range(8, w_bytes).expect("w range");
    refused(
        c,
        "w off its 16-byte alignment",
        "matvec_dq w range starts at offset 8, which is not a multiple of 16",
    );
    let mut c = call(&rig, w, x, y_ok);
    c.x = skewed_x.range(4, x_bytes).expect("x range");
    refused(
        c,
        "x off its 16-byte alignment",
        "matvec_dq x range starts at offset 4, which is not a multiple of 16",
    );
    let mut c = call(&rig, w, x, y_ok);
    c.y = y.range(2, y_bytes).expect("y range");
    refused(
        c,
        "y off its 4-byte alignment",
        "matvec_dq y range starts at offset 2, which is not a multiple of 4",
    );

    let mut c = call(&rig, w, x, y_ok);
    c.y = rig.w.range(0, y_bytes).expect("y range in w");
    refused(
        c,
        "y inside w",
        "matvec_dq y and matvec_dq w ranges overlap",
    );
    let mut c = call(&rig, w, x, y_ok);
    c.y = rig.x_dev.range(1024, y_bytes).expect("y range in x");
    refused(
        c,
        "y inside x",
        "matvec_dq y and matvec_dq x ranges overlap",
    );
}

/// The dq route's entry over a resident weight keeps the matvec's refusal: a weight of a type
/// the kernel does not serve is a `BadRequest` before any launch, and `y` keeps its zeros:
/// nothing runs in its place.
#[test]
fn matvec_dq_native_refuses_a_weight_the_kernel_does_not_serve() {
    if !testkit::device_present("matvec_dq_native_refuses_a_weight_the_kernel_does_not_serve") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (256, 4);
    let ty = QuantType::Q5K;
    let mut rng = testkit::Rng::new(0xD3_0002);
    let w = testkit::upload_bytes(&ctx, &reference::build_weight(ty, rows, k, &mut rng));
    let x = testkit::upload_f32(&ctx, &vec![0.0; k]);
    let y = testkit::upload_filled(&ctx, 0, rows * F32);
    let weight = NativeWeight::from_buffer(w, ty, k, rows, None).expect("a resident weight");
    assert_bad_request(
        ctx.matvec_dq_native(
            &stream,
            &weight,
            x.range(0, k * F32).expect("x range"),
            y.range(0, rows * F32).expect("y range"),
        ),
        "a resident weight of a type the kernel does not serve",
        "matvec_dq needs a Q4_K, Q6_K or Q8_0 weight",
    );
    stream.synchronize().expect("drain");
    let y = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * F32));
    assert!(y.iter().all(|&v| v == 0.0), "a launch wrote y: {y:?}");
}

/// Stage 2's decider on the first shape that made the dq route slow: k 4096, 12288 rows, Q6_K
/// (18 weights of the step have it). The one-column matvec and the new kernel on the same weight
/// in the same process, one sample of each interleaved with the first arm alternated, medians
/// of 100 after one dropped sample. Queued: 16 launches and one drain, per launch, which is the
/// kernel's own cost. Drained: one launch and one drain, which is what a drain point pays. A
/// microbenchmark, not a gate: it asserts nothing beyond the launch results. The Q4_K arm of
/// the same shape follows the Q6_K arm.
#[test]
#[ignore = "bench: exclusive timing slot"]
fn matvec_dq_first_shape_against_the_matvec_bench() {
    if !testkit::device_present("matvec_dq_first_shape_against_the_matvec_bench") {
        return;
    }
    for (ti, ty) in [QuantType::Q6K, QuantType::Q4K, QuantType::Q80]
        .into_iter()
        .enumerate()
    {
        first_shape_against_the_matvec(ty, ti as u64);
    }
}

/// The bench of one type on the first shape, in a context of its own.
fn first_shape_against_the_matvec(ty: QuantType, ti: u64) {
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (4096, 12288);
    let rig = rig(&ctx, ty, k, rows, 0xD0_0200 + ti);
    let w_bytes = ty.weight_bytes(rows, k).expect("size");
    let x_q8_bytes = q8_1_bytes(k, 1).expect("size");
    let x_q8 = testkit::upload_filled(&ctx, 0, x_q8_bytes);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: rig.x_dev.range(0, k * F32).expect("x"),
            y: x_q8.range(0, x_q8_bytes).expect("y"),
            k,
            ncols: 1,
        },
    )
    .expect("quantize");
    let y_bytes = rows * F32;
    let y_matvec = testkit::upload_filled(&ctx, 0, y_bytes);
    let y_dq = testkit::upload_filled(&ctx, 0, y_bytes);
    let matvec = || {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty,
                w: rig.w.range(0, w_bytes).expect("w range"),
                x_q8: x_q8.range(0, x_q8_bytes).expect("x range"),
                y: y_matvec.range(0, y_bytes).expect("y range"),
                k,
                rows,
                ncols: 1,
            },
        )
        .expect("matvec")
    };
    let dq = || {
        ctx.matvec_dq(
            &stream,
            call(
                &rig,
                rig.w.range(0, w_bytes).expect("w range"),
                rig.x_dev.range(0, k * F32).expect("x range"),
                y_dq.range(0, y_bytes).expect("y range"),
            ),
        )
        .expect("matvec_dq")
    };
    /// Microseconds a launch of `launch` takes: `queued` of them, the drain inside the sample.
    fn sample(stream: &DeviceStream, launch: impl Fn(), queued: usize) -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..queued {
            launch();
        }
        stream.synchronize().expect("drain");
        t.elapsed().as_secs_f64() * 1e6 / queued as f64
    }

    const SAMPLES: usize = 100;
    const QUEUED: usize = 16;
    let (mut queued, mut drained) = (Vec::new(), Vec::new());
    matvec();
    dq();
    stream.synchronize().expect("drain");
    for i in 0..=SAMPLES {
        let (mut q, mut d) = ([0f64; 2], [0f64; 2]);
        for arm in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
            let launch = || {
                if arm == 0 {
                    matvec();
                } else {
                    dq();
                }
            };
            q[arm] = sample(&stream, launch, QUEUED);
            d[arm] = sample(&stream, launch, 1);
        }
        if i > 0 {
            queued.push(q);
            drained.push(d);
        }
    }

    let sorted = |all: &[[f64; 2]], pick: fn(&[f64; 2]) -> f64| -> Vec<f64> {
        let mut v: Vec<f64> = all.iter().map(pick).collect();
        v.sort_by(|p, q| p.total_cmp(q));
        v
    };
    println!(
        "matvec dq first shape: {ty:?} k {k} rows {rows}, {w_bytes} weight bytes, {SAMPLES} interleaved samples"
    );
    for (mode, all) in [("queued", &queued), ("drained", &drained)] {
        let m = sorted(all, |s| s[0]);
        let d = sorted(all, |s| s[1]);
        let r = sorted(all, |s| s[1] / s[0]);
        let (mid, last) = (SAMPLES / 2, SAMPLES - 1);
        println!(
            "{mode}: matvec median {:.1} us (min {:.1}, max {:.1}), {:.0} GB/s; \
             dq median {:.1} us (min {:.1}, max {:.1}), {:.0} GB/s; \
             dq over matvec: median {:.3} (min {:.3}, max {:.3})",
            m[mid],
            m[0],
            m[last],
            w_bytes as f64 / m[mid] / 1e3,
            d[mid],
            d[0],
            d[last],
            w_bytes as f64 / d[mid] / 1e3,
            r[mid],
            r[0],
            r[last]
        );
    }
}
