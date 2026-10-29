//! Determinism of the dense projection shapes of one Qwen3.6-35B-A3B decode step: `pack` + `matmul`
//! at m = 1, 1,000 repeats per shape, every repeat byte-identical to the first. The weights come
//! from `testkit::upload_weight` and the activations are normal values, not the small integers of
//! `testkit::input`: those make every fp32 sum exact, which hides any change of reduction order.
//!
//! Shapes: `engine/src/family/spec.rs` `QWEN3_6_35B_A3B` and the real header table in
//! `family/gguf_model/inventory/tests.rs`. The tests assume `--test-threads=1`.

use super::{CudaContext, DeviceBuffer, DeviceStream, MatmulIo, PackCall, Projection, native};
use crate::kernels::cuda::testkit::{self, Mismatch, REPEATS};
use crate::kernels::cuda::{DeviceRange, TiledWeight};
use crate::weights::GgufType;

/// One decode projection: `rows` x `k` of type `ty`, and its K partials (see `k_partials`).
struct Shape {
    name: &'static str,
    ty: GgufType,
    rows: usize,
    k: usize,
    partials: usize,
}

const fn shape(name: &'static str, ty: GgufType, rows: usize, k: usize, partials: usize) -> Shape {
    Shape {
        name,
        ty,
        rows,
        k,
        partials,
    }
}

/// Every dense projection of a decode step.
const DENSE: [Shape; 7] = [
    shape("q_qkv", GgufType::Q80, 8192, 2048, 1),
    shape("kv_shared_gate_up", GgufType::Q80, 512, 2048, 1),
    shape("z_gate", GgufType::Q80, 4096, 2048, 1),
    shape("o_ssm_out", GgufType::Q80, 2048, 4096, 2),
    shape("alpha_beta", GgufType::Q80, 32, 2048, 1),
    shape("shared_down", GgufType::Q80, 2048, 512, 1),
    shape("lm_head", GgufType::Q6K, 248_320, 2048, 1),
];

/// The two expert shapes as (rows of one expert, k, partials); the device test of the expert
/// shapes keeps the same numbers in `resident/matmul_expert/tests/repeat.rs`.
const EXPERT: [(usize, usize, usize); 2] = [(512, 2048, 1), (2048, 512, 1)];

/// The number of K partials `mma884::split_k(rows / 8, k / 256)` picks. A copy of
/// `engine/vendor/xavier-qwen/src/mma884.cuh` line 118; the test below pins that line.
fn k_partials(rows: usize, k: usize) -> usize {
    let (warps, nb) = (rows / 8, k / 256);
    let mut split = 1;
    while warps * split < 4096 && nb / (split * 2) >= 8 {
        split *= 2;
    }
    split
}

/// Packs `x` into `xf` and multiplies all rows of `weight` into `y`, on `stream`.
fn queue_projection(
    ctx: &CudaContext,
    stream: &DeviceStream,
    run: (&TiledWeight, &DeviceBuffer),
    xf: &DeviceBuffer,
    y: DeviceRange<'_>,
) {
    let (weight, x) = run;
    let (rows, k) = (weight.n_rows(), weight.k());
    let need = native::activation_bytes(rows, k, 1).expect("sizes of the call");
    let pack = PackCall {
        x: x.range(0, need.x).expect("x range"),
        xf: xf.range(0, need.xf).expect("xf range"),
        m: 1,
        k,
    };
    ctx.pack(stream, pack).expect("queue the pack");
    let projection = Projection {
        weight,
        first_row: 0,
        rows,
    };
    let io = MatmulIo {
        xf: xf.range(0, need.xf).expect("xf range"),
        y,
        m: 1,
    };
    ctx.matmul(stream, projection, io)
        .expect("queue the matmul");
}

/// m = 1 activations: normal values from the seed, as f32 (the pack rounds them to fp16).
fn activations(seed: u64, k: usize) -> Vec<f32> {
    let mut rng = testkit::Rng::new(seed);
    (0..k).map(|_| rng.normal() as f32).collect()
}

/// A y buffer of `need.y` bytes that holds 7.0 in every element.
fn stale_y(ctx: &CudaContext, bytes: usize) -> DeviceBuffer {
    let pattern = 7.0f32.to_le_bytes();
    let stale: Vec<u8> = pattern.into_iter().cycle().take(bytes).collect();
    testkit::upload_bytes(ctx, &stale)
}

/// Repeats one shape on stream A; at repeat 50 stream B runs the same work on its own stream,
/// overlapped with A's chunk, and its output must equal A's.
fn repeat_dense(ctx: &CudaContext, index: usize, shape: &Shape) {
    let Shape {
        name, ty, rows, k, ..
    } = *shape;
    let weight = testkit::upload_weight(ctx, ty, rows, k);
    let x = testkit::upload_f32(ctx, &activations(0x52D2_E000 + index as u64, k));
    let need = native::activation_bytes(rows, k, 1).expect("sizes of the call");
    assert_eq!(need.y, rows * 4, "{name}: y holds one f32 per row");
    let (a, b) = (
        ctx.stream().expect("stream A"),
        ctx.stream().expect("stream B"),
    );
    let (xf_a, xf_b) = (
        ctx.allocate(need.xf).expect("xf of stream A"),
        ctx.allocate(need.xf).expect("xf of stream B"),
    );
    let y_b = stale_y(ctx, need.y);
    let run = (&weight, &x);
    let want = testkit::run_repeats(ctx, &a, need.y, |repeat, y| {
        queue_projection(ctx, &a, run, &xf_a, y);
        if repeat == 50 {
            let y_b = y_b.range(0, need.y).expect("y range of stream B");
            queue_projection(ctx, &b, run, &xf_b, y_b);
        }
    })
    .unwrap_or_else(|Mismatch { repeat, index }| {
        panic!(
            "{name}: repeat {repeat} differs from repeat 0 at byte {index} (element {})",
            index / 4
        )
    });
    b.synchronize().expect("drain stream B");
    let mut got = vec![0u8; need.y];
    y_b.read_at(&b, 0, &mut got).expect("read stream B");
    assert_eq!(
        testkit::first_difference(&want, &got),
        None,
        "{name}: the second stream differs from the first"
    );
    println!(
        "projection_repeat: {name} {ty:?} {rows}x{k} repeats={REPEATS} y_bytes={} second_stream=equal",
        need.y
    );
}

#[test]
fn projection_repeat_dense_shapes_are_bitwise_equal() {
    let test = "projection_repeat_dense_shapes_are_bitwise_equal";
    if !testkit::device_present(test) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    for (index, shape) in DENSE.iter().enumerate() {
        repeat_dense(&ctx, index, shape);
    }
}

#[test]
fn projection_repeat_compare_reports_the_first_differing_index() {
    let first = testkit::first_difference;
    assert_eq!(first(&[1, 2, 3], &[1, 2, 3]), None);
    assert_eq!(first(&[9, 2, 3], &[1, 2, 3]), Some(0));
    assert_eq!(first(&[1, 2, 3], &[1, 2, 9]), Some(2));
    assert_eq!(first(&[1, 9, 3, 9], &[1, 2, 3, 2]), Some(1));
    assert_eq!(first(&[1, 2, 3, 4], &[1, 2, 3, 4, 5]), Some(4));
    let nan = 0x7fc0_0000u32.to_le_bytes();
    let other_nan = 0x7fc0_0001u32.to_le_bytes();
    assert_eq!(first(&nan, &other_nan), Some(0));
}

#[test]
fn projection_repeat_catches_a_perturbed_input() {
    let test = "projection_repeat_catches_a_perturbed_input";
    if !testkit::device_present(test) {
        return;
    }
    let shape = &DENSE[4];
    let Shape { ty, rows, k, .. } = *shape;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = testkit::upload_weight(&ctx, ty, rows, k);
    let values = activations(0x52D2_E004, k);
    let x = testkit::upload_f32(&ctx, &values);
    let need = native::activation_bytes(rows, k, 1).expect("sizes of the call");
    let xf = ctx.allocate(need.xf).expect("xf");
    let perturbed = (values[0] + 1.0).to_le_bytes();
    let result = testkit::run_repeats(&ctx, &stream, need.y, |repeat, y| {
        if repeat == 537 {
            x.write_at(&stream, 0, &perturbed).expect("perturb x");
        }
        queue_projection(&ctx, &stream, (&weight, &x), &xf, y);
    });
    let Err(Mismatch { repeat, index }) = result else {
        panic!("a changed input element went unseen");
    };
    assert_eq!(repeat, 537);
    assert!(index < need.y, "the index {index} lies inside y");
}

#[test]
fn projection_repeat_shapes_have_at_most_two_k_partials() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/vendor/xavier-qwen/src/mma884.cuh"
    ));
    let rule = "while (warps * split < 4096 && nb / (split * 2) >= 8) split *= 2;";
    assert!(
        source.lines().any(|line| line.trim() == rule),
        "the vendored split_k rule changed; update k_partials and re-check the 2-partial claim"
    );
    let experts = EXPERT
        .iter()
        .map(|&(rows, k, partials)| ("expert", rows, k, partials));
    let dense = DENSE.iter().map(|s| (s.name, s.rows, s.k, s.partials));
    for (name, rows, k, partials) in dense.chain(experts) {
        assert_eq!(k_partials(rows, k), partials, "{name}: {rows}x{k}");
        assert!(partials <= 2, "{name}: more than two K partials");
    }
}
