//! Device tests (3090) of `matvec_expert` (seam 1: the Rust wrapper against the C entry against
//! the trimmed vendored `mul_mat_vec_q` ids path). Device tests assume `--test-threads=1`.
//!
//! The host reference is the one of `matvec_q`'s tests (`reference`), shared so the two files
//! cannot drift: the dense per-slot oracle of test 5 is `matvec_q` itself on the selected
//! expert's weights.

use super::{ExpertMatvecCall, ExpertShape, expert_needs};
use crate::EngError;
use crate::kernels::cuda::resident::matvec_q::tests::{
    Owner, assert_bad_request, bench, nan_filled, reference,
};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceStream};

#[allow(unsafe_code)]
unsafe extern "C" {
    /// the vendored mmvq.cu test-only arm override for INFERENCE_CUDA_MVQ_WPR.
    safe fn inference_cuda_test_arm_mvq_wpr(on: i32);
}

const N_EXPERT: usize = 16;
const N_USED: usize = 8;
const IDS: [i32; N_USED] = [0, 15, 3, 3, 7, 15, 1, 9];
const TYPES: [QuantType; 4] = [
    QuantType::Q4K,
    QuantType::Q50,
    QuantType::Q6K,
    QuantType::Q80,
];
const GATE_UP: (usize, usize) = (2048, 512);
const DOWN: (usize, usize) = (512, 2048);
/// k 2304 pads to 2560: the per-slot activation rows are 80 q8_1 blocks apart, not 72 (the other
/// shapes have k % 512 == 0, where the two strides coincide).
const PADDED: (usize, usize) = (2304, 64);
const NAN_BITS: u32 = 0x7fc0_0000;

fn ids_buffer(ctx: &CudaContext, ids: &[i32]) -> DeviceBuffer {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    testkit::upload_bytes(ctx, &bytes)
}

/// Builds and uploads a stack and the quantized activations, both x_per_slot modes sharing the
/// seed; returns (stack, ids, shared x, per-slot x).
fn stack_rig(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    (k, rows): (usize, usize),
    seed: u64,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let mut rng = testkit::Rng::new(seed);
    let w_bytes = reference::build_weight(ty, rows * N_EXPERT, k, &mut rng);
    let stack = testkit::upload_bytes(ctx, &w_bytes);
    let ids = ids_buffer(ctx, &IDS);
    let mut quantize = |ncols: usize| {
        let x: Vec<f32> = (0..ncols * k)
            .map(|_| (rng.normal() as f32) * 0.05)
            .collect();
        let x_dev = testkit::upload_f32(ctx, &x);
        let bytes = q8_1_bytes(k, ncols).expect("size");
        let out = testkit::upload_filled(ctx, 0xAA, bytes);
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: x_dev.range(0, x.len() * 4).expect("x"),
                y: out.range(0, bytes).expect("y"),
                k,
                ncols,
            },
        )
        .expect("quantize");
        stream.synchronize().expect("drain");
        out
    };
    let shared = quantize(1);
    let per_slot = quantize(N_USED);
    (stack, ids, shared, per_slot)
}

/// The dense oracle: `matvec_q` on expert `id`'s weights against activation row `col`.
#[allow(
    clippy::too_many_arguments,
    reason = "a test helper that spells out the oracle's view"
)]
fn dense_slot(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    stack: &DeviceBuffer,
    x_q8: &DeviceBuffer,
    (k, rows): (usize, usize),
    id: usize,
    col: usize,
) -> Vec<u8> {
    let y_bytes = rows * 4;
    let y = testkit::upload_filled(ctx, 0xAA, y_bytes);
    let row_bytes = q8_1_bytes(k, 1).expect("size");
    ctx.matvec_q(
        stream,
        MatvecCall {
            ty,
            w: stack
                .range(
                    id * ty.weight_bytes(rows, k).expect("size"),
                    ty.weight_bytes(rows, k).expect("size"),
                )
                .expect("expert w"),
            x_q8: x_q8.range(col * row_bytes, row_bytes).expect("x col"),
            y: y.range(0, y_bytes).expect("y"),
            k,
            rows,
            ncols: 1,
        },
    )
    .expect("dense oracle");
    stream.synchronize().expect("drain");
    testkit::read_bytes(ctx, &y, y_bytes)
}

#[allow(
    clippy::too_many_arguments,
    reason = "a test helper that spells out the launch view"
)]
fn expert_launch(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    stack: &DeviceBuffer,
    ids: &DeviceBuffer,
    x_q8: &DeviceBuffer,
    y: &DeviceBuffer,
    y_offset: usize,
    (k, rows): (usize, usize),
    per_slot: bool,
    y_stride: usize,
) {
    ctx.matvec_expert(
        stream,
        ExpertMatvecCall {
            ty,
            w_stack: stack
                .range(0, ty.weight_bytes(rows * N_EXPERT, k).expect("size"))
                .expect("stack"),
            ids: ids.range(0, N_USED * 4).expect("ids"),
            x_q8: x_q8
                .range(
                    0,
                    q8_1_bytes(k, if per_slot { N_USED } else { 1 }).expect("size"),
                )
                .expect("x"),
            x_per_slot: per_slot,
            y: y.range(y_offset, (y_stride * (N_USED - 1) + rows) * 4)
                .expect("y"),
            y_slot_stride: y_stride,
            k,
            rows,
            n_expert: N_EXPERT,
            n_used: N_USED,
            glu_gate: None,
        },
    )
    .expect("expert matvec");
}

#[test]
#[ignore = "requires native CUDA; run explicitly under the GPU lease"]
fn mapped_expert_weights_match_device_weights() {
    assert!(testkit::device_present(
        "mapped_expert_weights_match_device_weights"
    ));
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let shape = (512, 64);
    for (ty, gguf) in [
        (QuantType::Q4K, crate::weights::GgufType::Q4K),
        (QuantType::Q6K, crate::weights::GgufType::Q6K),
    ] {
        let seed = 0x9123_4567;
        let mut rng = testkit::Rng::new(seed);
        let bytes = reference::build_weight(ty, shape.1 * N_EXPERT, shape.0, &mut rng);
        let (device, ids, shared, per_slot) = stack_rig(&ctx, &stream, ty, shape, seed);
        let tensor = crate::kernels::cuda::QuantTensor {
            name: "experts",
            ty: gguf,
            n_rows: shape.1 * N_EXPERT,
            k: shape.0,
            source_len: bytes.len(),
        };
        let pinned_before = ctx.live_owners().pinned;
        let device_before = ctx.live_bytes();
        let failed = ctx.upload_native_mapped(tensor, 4096, &mut |_, _| {
            Err(EngError::Checkpoint("injected read failure".into()))
        });
        assert!(matches!(failed, Err(EngError::Checkpoint(_))));
        assert_eq!(
            ctx.live_owners().pinned,
            pinned_before,
            "failed read released mapping"
        );
        let mapped = ctx
            .upload_native_mapped(tensor, 4096, &mut |at, out| {
                out.copy_from_slice(&bytes[at..at + out.len()]);
                Ok(())
            })
            .expect("mapped upload");
        assert_eq!(ctx.live_owners().pinned, pinned_before + bytes.len());
        assert_eq!(
            ctx.live_bytes(),
            device_before,
            "mapped weights do not consume the device ledger"
        );
        for (per, x) in [(false, &shared), (true, &per_slot)] {
            let n = shape.1 * N_USED * 4;
            let actual = testkit::upload_filled(&ctx, 0xAA, n);
            let expected = testkit::upload_filled(&ctx, 0xAA, n);
            expert_launch(
                &ctx, &stream, ty, &device, &ids, x, &expected, 0, shape, per, shape.1,
            );
            expert_launch(
                &ctx,
                &stream,
                ty,
                mapped.buffer(),
                &ids,
                x,
                &actual,
                0,
                shape,
                per,
                shape.1,
            );
            stream.synchronize().expect("drain");
            assert_eq!(
                testkit::read_bytes(&ctx, &actual, n),
                testkit::read_bytes(&ctx, &expected, n),
                "{ty:?} per_slot={per}"
            );
        }
        drop(mapped);
        assert_eq!(
            ctx.live_owners().pinned,
            pinned_before,
            "mapping released after use"
        );
    }
}

#[test]
fn expert_needs_are_pure_and_exact() {
    let shape = ExpertShape {
        ty: QuantType::Q4K,
        k: 2048,
        rows: 512,
        n_expert: 16,
        n_used: 8,
        x_per_slot: false,
        y_slot_stride: 1024,
    };
    let need = expert_needs(&shape).expect("sizes");
    // 16 experts of 512 rows of 8 blocks; 8 ids; one activation row; (8 - 1) * 1024 + 512 floats.
    assert_eq!(
        (need.w_stack, need.ids, need.x_q8, need.y),
        (16 * 512 * 8 * 144, 8 * 4, 64 * 36, (7 * 1024 + 512) * 4)
    );
    let per_slot = ExpertShape {
        x_per_slot: true,
        ..shape
    };
    assert_eq!(expert_needs(&per_slot).expect("sizes").x_q8, 8 * 64 * 36);
    let packed = ExpertShape {
        y_slot_stride: 512,
        ..shape
    };
    assert_eq!(expert_needs(&packed).expect("sizes").y, (7 * 512 + 512) * 4);
    // k 2304 pads to 2560: the per-slot activation rows are 80 blocks, the stack rows 9 K-blocks.
    let padded = ExpertShape {
        k: 2304,
        rows: 64,
        x_per_slot: true,
        ..shape
    };
    let need = expert_needs(&padded).expect("sizes");
    assert_eq!((need.w_stack, need.x_q8), (16 * 64 * 9 * 144, 8 * 80 * 36));
    // Overflows are None, not a panic or a wrapped size: rows * n_expert, the y footprint, and a
    // shape without used slots.
    let huge_stack = ExpertShape {
        rows: usize::MAX / 2,
        n_expert: 4,
        ..shape
    };
    assert!(expert_needs(&huge_stack).is_none());
    let huge_stride = ExpertShape {
        y_slot_stride: usize::MAX,
        ..shape
    };
    assert!(expert_needs(&huge_stride).is_none());
    let no_slots = ExpertShape { n_used: 0, ..shape };
    assert!(expert_needs(&no_slots).is_none());
    // A `rows * n_expert` that WRAPS TO ZERO while every later product stays small, so only the
    // checked multiply refuses it. rows 1<<61 x 8 experts wraps to 0 (a wrapping form would
    // answer weight_bytes(0, k) = 0 and pass), and the y footprint (0 * stride + 1<<61, times
    // 4 = 1<<63) still fits usize. The 1<<62 x 4 form wraps as well; BOTH forms are refused by
    // the rows.checked_mul(n_expert) multiply itself (1<<62 * 4 = 2^64 overflows it too — the
    // second form's y footprint would also overflow its own multiply, but the w_stack field is
    // evaluated first); both are pinned.
    let wrap_zero = ExpertShape {
        rows: 1 << 61,
        n_expert: 8,
        n_used: 1,
        y_slot_stride: 1 << 61,
        ..shape
    };
    assert!(expert_needs(&wrap_zero).is_none());
    let wrap_zero_y_too = ExpertShape {
        rows: 1 << 62,
        n_expert: 4,
        ..shape
    };
    assert!(expert_needs(&wrap_zero_y_too).is_none());
}

#[test]
fn matvec_expert_equals_the_dense_launch_per_slot() {
    if !testkit::device_present("matvec_expert_equals_the_dense_launch_per_slot") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (si, shape) in [GATE_UP, DOWN, PADDED].iter().enumerate() {
        for (ti, ty) in TYPES.iter().enumerate() {
            let seed = 0x8000_0000 + si as u64 * 1000 + ti as u64;
            let (stack, ids, shared, per_slot) = stack_rig(&ctx, &stream, *ty, *shape, seed);
            // y is [slot][gate|up]: one buffer of stride 2 * rows, the gate launch writes
            // offset 0 and the up launch offset rows of each slot.
            let (_, rows) = *shape;
            let stride = 2 * rows;
            // The full [slot][gate|up] buffer: every slot has both halves, so the canary
            // check can look at the up half of the last slot too.
            let y_bytes = stride * N_USED * 4;
            let gate_y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
            expert_launch(
                &ctx, &stream, *ty, &stack, &ids, &shared, &gate_y, 0, *shape, false, stride,
            );
            stream.synchronize().expect("drain");
            let gate_bytes = testkit::read_bytes(&ctx, &gate_y, y_bytes);
            for (slot, id) in IDS.iter().enumerate() {
                let want = dense_slot(&ctx, &stream, *ty, &stack, &shared, *shape, *id as usize, 0);
                let got = &gate_bytes[slot * stride * 4..slot * stride * 4 + rows * 4];
                assert_eq!(
                    got,
                    want.as_slice(),
                    "{ty:?} {shape:?} slot {slot} (gate, shared x)"
                );
                // The up half of every slot is still the fill: the gate launch wrote only rows.
                for byte in &gate_bytes[slot * stride * 4 + rows * 4..(slot + 1) * stride * 4] {
                    assert_eq!(*byte, 0xAA, "{ty:?} gate launch wrote into the up half");
                }
            }
            let up_y = testkit::upload_filled(&ctx, 0x55, y_bytes);
            expert_launch(
                &ctx,
                &stream,
                *ty,
                &stack,
                &ids,
                &per_slot,
                &up_y,
                rows * 4,
                *shape,
                true,
                stride,
            );
            stream.synchronize().expect("drain");
            let up_bytes = testkit::read_bytes(&ctx, &up_y, y_bytes);
            for (slot, id) in IDS.iter().enumerate() {
                let want = dense_slot(
                    &ctx,
                    &stream,
                    *ty,
                    &stack,
                    &per_slot,
                    *shape,
                    *id as usize,
                    slot,
                );
                let got = &up_bytes[slot * stride * 4 + rows * 4..slot * stride * 4 + 2 * rows * 4];
                assert_eq!(
                    got,
                    want.as_slice(),
                    "{ty:?} {shape:?} slot {slot} (up, per-slot x)"
                );
            }
        }
    }
}

#[test]
fn matvec_expert_follows_the_device_ids() {
    if !testkit::device_present("matvec_expert_follows_the_device_ids") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ty = QuantType::Q4K;
    let shape = GATE_UP;
    let (_, rows) = shape;
    let (stack, ids, shared, _) = stack_rig(&ctx, &stream, ty, shape, 0x8100_0001);
    let y_bytes = rows * N_USED * 4;
    let eager = |ids: &DeviceBuffer| {
        let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
        expert_launch(
            &ctx, &stream, ty, &stack, ids, &shared, &y, 0, shape, false, rows,
        );
        stream.synchronize().expect("drain");
        testkit::read_bytes(&ctx, &y, y_bytes)
    };
    // Two eager launches with the ids rewritten in between follow the new word.
    let before = eager(&ids);
    let rotated: [i32; N_USED] = [9, 1, 15, 7, 3, 3, 0, 15];
    let ids2 = ids_buffer(&ctx, &rotated);
    let after = eager(&ids2);
    for (slot, id) in rotated.iter().enumerate() {
        let want = dense_slot(&ctx, &stream, ty, &stack, &shared, shape, *id as usize, 0);
        assert_eq!(
            &after[slot * rows * 4..(slot + 1) * rows * 4],
            want.as_slice(),
            "eager slot {slot} did not follow the new id"
        );
        assert_ne!(
            &before[slot * rows * 4..(slot + 1) * rows * 4],
            want.as_slice(),
            "slot {slot} id did not change"
        );
    }
    // Captured: the replay follows a rewritten ids word, one kernel node.
    let cap_y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let launch = || -> Result<(), crate::EngError> {
        expert_launch(
            &ctx, &stream, ty, &stack, &ids2, &shared, &cap_y, 0, shape, false, rows,
        );
        Ok(())
    };
    let (graph, ()) = ctx.capture(&stream, plan, launch).expect("capture");
    assert_eq!(graph.nodes(), 1, "one expert kernel node");
    // The captured graph reads the ids from ids2's address, so the replay follows words
    // rewritten into that same buffer.
    let rotated_again: [i32; N_USED] = [15, 0, 1, 9, 7, 15, 3, 3];
    let rotated_bytes: Vec<u8> = rotated_again
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    ids2.write_at(&stream, 0, &rotated_bytes)
        .expect("rewrite ids");
    graph.launch(&stream).expect("replay");
    stream.synchronize().expect("drain");
    let replay = testkit::read_bytes(&ctx, &cap_y, y_bytes);
    for (slot, id) in rotated_again.iter().enumerate() {
        let want = dense_slot(&ctx, &stream, ty, &stack, &shared, shape, *id as usize, 0);
        assert_eq!(
            &replay[slot * rows * 4..(slot + 1) * rows * 4],
            want.as_slice(),
            "replay slot {slot} did not follow the rewritten id"
        );
    }
}

fn check_out_of_range_ids(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    shape: (usize, usize),
    (per_slot, seed): (bool, u64),
) {
    let (_, rows) = shape;
    let (stack, _, shared, per_slot_x) = stack_rig(ctx, stream, ty, shape, seed);
    let x = if per_slot { &per_slot_x } else { &shared };
    // The router writes -1 only on a faulted row; n_expert itself is the other edge.
    let bad: [i32; N_USED] = [-1, 4, 15, N_EXPERT as i32, 2, 8, 0, 11];
    let ids = ids_buffer(ctx, &bad);
    let y_bytes = rows * N_USED * 4;
    let y = testkit::upload_filled(ctx, 0xAA, y_bytes);
    expert_launch(
        ctx, stream, ty, &stack, &ids, x, &y, 0, shape, per_slot, rows,
    );
    stream.synchronize().expect("drain");
    let got = testkit::read_bytes(ctx, &y, y_bytes);
    for (slot, id) in bad.iter().enumerate() {
        let slot_bytes = &got[slot * rows * 4..(slot + 1) * rows * 4];
        if *id < 0 || *id as usize >= N_EXPERT {
            for (wi, word) in slot_bytes.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(
                    u32::from_le_bytes(*word),
                    NAN_BITS,
                    "{ty:?} {shape:?} slot {slot} (id {id}) row {wi} is not the quiet NaN"
                );
            }
        } else {
            let col = if per_slot { slot } else { 0 };
            let want = dense_slot(ctx, stream, ty, &stack, x, shape, *id as usize, col);
            assert_eq!(
                slot_bytes,
                want.as_slice(),
                "{ty:?} {shape:?} valid slot {slot} changed"
            );
        }
    }
}

/// The gate/up shape (k 2048, one row per block) and the production Q4_K down shape (k 512, 2048
/// rows, per-slot x): its K loop is shorter than one block iteration, so the launch is the
/// small-k one with several rows per block and the NaN loop of the guard writes several rows
/// there (a loop that writes row 0 only fails).
#[test]
fn matvec_expert_ids_out_of_range_are_documented() {
    if !testkit::device_present("matvec_expert_ids_out_of_range_are_documented") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    check_out_of_range_ids(&ctx, &stream, QuantType::Q6K, GATE_UP, (false, 0x8200_0001));
    check_out_of_range_ids(&ctx, &stream, QuantType::Q4K, DOWN, (true, 0x8200_0002));
}

#[derive(Clone, Copy)]
enum YIn {
    Own,
    Stack,
    Ids,
    X,
}

#[derive(Clone, Copy)]
struct Spec {
    k: usize,
    rows: usize,
    n_expert: usize,
    n_used: usize,
    y_slot_stride: usize,
    x_per_slot: bool,
    stack: (usize, usize),
    ids: (usize, usize),
    x: (usize, usize),
    y: (usize, usize),
    y_in: YIn,
    stack_of: Owner,
    ids_of: Owner,
    stream_of: Owner,
}

/// Q4_K, k 768 (three blocks, padded to 1024), 16 experts of 64 rows, 8 used slots, the
/// `[slot][gate|up]` stride of 128 floats and one shared activation row.
const STACK_NEED: usize = 16 * 64 * 3 * 144;
const IDS_NEED: usize = 8 * 4;
const X_ROW: usize = 32 * 36;
const Y_NEED: usize = (7 * 128 + 64) * 4;
const VALID: Spec = Spec {
    k: 768,
    rows: 64,
    n_expert: 16,
    n_used: 8,
    y_slot_stride: 128,
    x_per_slot: false,
    stack: (0, STACK_NEED),
    ids: (0, IDS_NEED),
    x: (0, X_ROW),
    y: (0, Y_NEED),
    y_in: YIn::Own,
    stack_of: Owner::Own,
    ids_of: Owner::Own,
    stream_of: Owner::Own,
};

/// The rig buffers are larger than the call needs, so a case can offset, shorten or overlap a range.
const STACK_BUFFER: usize = STACK_NEED + 8192;
const SMALL_BUFFER: usize = 16_384;

struct RefusalRig {
    stream: DeviceStream,
    stack: DeviceBuffer,
    ids: DeviceBuffer,
    x: DeviceBuffer,
    y: DeviceBuffer,
    ctx: CudaContext,
}

struct OtherContext {
    stream: DeviceStream,
    stack: DeviceBuffer,
    ids: DeviceBuffer,
    _ctx: CudaContext,
}

fn padded_ids(ctx: &CudaContext, ids: &[i32], total: usize) -> DeviceBuffer {
    let mut bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    bytes.resize(total, 0);
    testkit::upload_bytes(ctx, &bytes)
}

fn attempt(rig: &RefusalRig, other: &OtherContext, spec: Spec) -> Result<(), EngError> {
    let stack_buffer = match spec.stack_of {
        Owner::Own => &rig.stack,
        Owner::Other => &other.stack,
    };
    let ids_buffer = match spec.ids_of {
        Owner::Own => &rig.ids,
        Owner::Other => &other.ids,
    };
    let y_buffer = match spec.y_in {
        YIn::Own => &rig.y,
        YIn::Stack => &rig.stack,
        YIn::Ids => &rig.ids,
        YIn::X => &rig.x,
    };
    let stream = match spec.stream_of {
        Owner::Own => &rig.stream,
        Owner::Other => &other.stream,
    };
    rig.ctx.matvec_expert(
        stream,
        ExpertMatvecCall {
            ty: QuantType::Q4K,
            w_stack: stack_buffer.range(spec.stack.0, spec.stack.1)?,
            ids: ids_buffer.range(spec.ids.0, spec.ids.1)?,
            x_q8: rig.x.range(spec.x.0, spec.x.1)?,
            x_per_slot: spec.x_per_slot,
            y: y_buffer.range(spec.y.0, spec.y.1)?,
            y_slot_stride: spec.y_slot_stride,
            k: spec.k,
            rows: spec.rows,
            n_expert: spec.n_expert,
            n_used: spec.n_used,
            glu_gate: None,
        },
    )
}

#[test]
fn matvec_expert_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("matvec_expert_rejects_unsound_calls_before_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let rig = RefusalRig {
        stream: ctx.stream().expect("stream"),
        stack: testkit::upload_filled(&ctx, 0, STACK_BUFFER),
        ids: padded_ids(&ctx, &[3, 15, 0, 9, 1, 2, 4, 5], 8192),
        x: testkit::upload_filled(&ctx, 0, SMALL_BUFFER),
        y: testkit::upload_filled(&ctx, 0xAB, 8192),
        ctx,
    };
    let other_ctx = CudaContext::new().expect("second context");
    let other = OtherContext {
        stream: other_ctx.stream().expect("stream of the other context"),
        stack: testkit::upload_filled(&other_ctx, 0, STACK_BUFFER),
        ids: padded_ids(&other_ctx, &[3, 15, 0, 9, 1, 2, 4, 5], 8192),
        _ctx: other_ctx,
    };
    let cases: [(&str, &str, Spec); 31] = [
        (
            "k off the Q4_K grid",
            "needs k as a multiple of",
            Spec { k: 700, ..VALID },
        ),
        ("k 0", "needs k as a multiple of", Spec { k: 0, ..VALID }),
        (
            "no rows",
            "rows and experts above zero",
            Spec { rows: 0, ..VALID },
        ),
        (
            "no experts",
            "rows and experts above zero",
            Spec {
                n_expert: 0,
                ..VALID
            },
        ),
        (
            "rows 62",
            "needs rows as a multiple of",
            Spec { rows: 62, ..VALID },
        ),
        (
            "no used slots",
            "used slots, got 0",
            Spec { n_used: 0, ..VALID },
        ),
        (
            "nine used slots",
            "used slots, got 9",
            Spec { n_used: 9, ..VALID },
        ),
        (
            "more used slots than experts",
            "takes 1 to 4 used slots",
            Spec {
                n_expert: 4,
                ..VALID
            },
        ),
        (
            "y stride below rows",
            "is below the 64 rows",
            Spec {
                y_slot_stride: 63,
                ..VALID
            },
        ),
        (
            "rows * n_expert overflows",
            "size overflows",
            Spec {
                rows: (usize::MAX / 8) & !3,
                y_slot_stride: (usize::MAX / 8) & !3,
                ..VALID
            },
        ),
        (
            "stack one block short",
            "w_stack range holds",
            Spec {
                stack: (0, STACK_NEED - 144),
                ..VALID
            },
        ),
        (
            "stack misaligned by 2",
            "w_stack range starts at offset",
            Spec {
                stack: (2, STACK_NEED),
                ..VALID
            },
        ),
        (
            "ids one int short",
            "ids range holds",
            Spec {
                ids: (0, IDS_NEED - 4),
                ..VALID
            },
        ),
        (
            "ids misaligned by 2",
            "ids range starts at offset",
            Spec {
                ids: (2, IDS_NEED),
                ..VALID
            },
        ),
        (
            "x one byte short",
            "x_q8 range holds",
            Spec {
                x: (0, X_ROW - 1),
                ..VALID
            },
        ),
        (
            "x unpadded: 24 blocks",
            "x_q8 range holds",
            Spec {
                x: (0, 24 * 36),
                ..VALID
            },
        ),
        (
            "per-slot x with the shared size",
            "x_q8 range holds",
            Spec {
                x_per_slot: true,
                ..VALID
            },
        ),
        (
            "x misaligned by 1",
            "x_q8 range starts at offset",
            Spec {
                x: (1, X_ROW),
                ..VALID
            },
        ),
        (
            "y without the rows of the last slot (A2)",
            "y range holds",
            Spec {
                y: (0, 7 * 128 * 4),
                ..VALID
            },
        ),
        (
            "y one float short",
            "y range holds",
            Spec {
                y: (0, Y_NEED - 4),
                ..VALID
            },
        ),
        (
            "y packed to rows instead of the stride",
            "y range holds",
            Spec {
                y: (0, 8 * 64 * 4),
                ..VALID
            },
        ),
        (
            "y misaligned by 2",
            "y range starts at offset",
            Spec {
                y: (2, Y_NEED),
                ..VALID
            },
        ),
        (
            "y over the stack",
            "y and w_stack ranges overlap",
            Spec {
                y: (4096, Y_NEED),
                y_in: YIn::Stack,
                ..VALID
            },
        ),
        (
            "y over the end of the stack",
            "y and w_stack ranges overlap",
            Spec {
                y: (STACK_NEED - 4, Y_NEED),
                y_in: YIn::Stack,
                ..VALID
            },
        ),
        (
            "y over the ids",
            "y and ids ranges overlap",
            Spec {
                y: (16, Y_NEED),
                y_in: YIn::Ids,
                ..VALID
            },
        ),
        (
            "y over the last id word",
            "y and ids ranges overlap",
            Spec {
                y: (28, Y_NEED),
                y_in: YIn::Ids,
                ..VALID
            },
        ),
        (
            "y over x",
            "y and x_q8 ranges overlap",
            Spec {
                y: (1024, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "y over the seventh per-slot activation row",
            "y and x_q8 ranges overlap",
            Spec {
                x_per_slot: true,
                x: (0, 8 * X_ROW),
                y: (7 * X_ROW, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "stack of another context",
            "buffer belongs",
            Spec {
                stack_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "ids of another context",
            "buffer belongs",
            Spec {
                ids_of: Owner::Other,
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
    rig.stream.synchronize().expect("drain");
    let untouched = testkit::read_bytes(&rig.ctx, &rig.y, 8192);
    assert!(untouched.iter().all(|&byte| byte == 0xAB), "y untouched");
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    attempt(&rig, &other, VALID).expect("the valid call launches");
    rig.stream.synchronize().expect("drain");
    let written = testkit::read_bytes(&rig.ctx, &rig.y, Y_NEED);
    let words: Vec<u32> = written
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    if let Err(why) = check_stride_layout(&words, VALID.n_used, VALID.y_slot_stride, VALID.rows) {
        panic!("{why}");
    }
}

/// The poison word of the unsound-call rig (`y` is filled with 0xAB bytes).
const POISON: u32 = 0xABAB_ABAB;

/// Footprint check: the `rows` live f32 words of every slot differ from the poison word and
/// the `stride - rows` gap words between slots keep it. Pure, so its own test can prove it is
/// not vacuous (`the_stride_layout_check_catches_an_unwritten_slot_and_an_overwritten_gap`).
fn check_stride_layout(
    words: &[u32],
    n_used: usize,
    stride: usize,
    rows: usize,
) -> Result<(), String> {
    for slot in 0..n_used {
        let first = slot * stride;
        let live = words.get(first..first + rows).ok_or_else(|| {
            format!(
                "slot {slot}: the live rows lie outside the {} words",
                words.len()
            )
        })?;
        if live.contains(&POISON) {
            return Err(format!(
                "the valid call left a live row of slot {slot} unwritten"
            ));
        }
        let gap = words
            .get(first + rows..(first + stride).min(words.len()))
            .unwrap_or(&[]);
        if gap.iter().any(|&word| word != POISON) {
            return Err(format!(
                "the valid call overwrote the stride gap of slot {slot}"
            ));
        }
    }
    Ok(())
}

#[test]
fn the_stride_layout_check_catches_an_unwritten_slot_and_an_overwritten_gap() {
    let (n_used, stride, rows) = (3, 8, 5);
    let mut good = vec![POISON; (n_used - 1) * stride + rows];
    for slot in 0..n_used {
        for row in 0..rows {
            good[slot * stride + row] = 0x3F80_0000 + (slot * rows + row) as u32;
        }
    }
    assert_eq!(check_stride_layout(&good, n_used, stride, rows), Ok(()));
    let mut unwritten = good.clone();
    unwritten[stride + 2] = POISON;
    assert!(
        check_stride_layout(&unwritten, n_used, stride, rows)
            .unwrap_err()
            .contains("slot 1 unwritten")
    );
    let mut overwritten = good.clone();
    overwritten[rows] = 0x3F80_0000;
    assert!(
        check_stride_layout(&overwritten, n_used, stride, rows)
            .unwrap_err()
            .contains("stride gap of slot 0")
    );
    assert!(check_stride_layout(&good[..stride], n_used, stride, rows).is_err());
}

/// The expert matvec bandwidth bench (timing only; run on the device with `--ignored`).
///
/// The serve section marks attributed ~35 GB/s to the expert matvec against ~82 GB/s for the
/// dense one, but the dense bench shapes are Q8_0 while the expert stacks are Q4_K, so the pair
/// mixed kernel types. This bench measures the REAL moe shapes - Q4_K gate/up (k 2048, rows 512,
/// shared x) and down (k 512, rows 2048, per-slot x), 8 of 16 experts - against a dense Q4_K
/// launch reading the same weight bytes, so the only difference is the type and the ids scatter.
/// GB/s counts the weight bytes the launch reads (n_used x rows x row bytes); cold = the launch
/// alone right after a 68 MB dense matvec that evicts the L2, which is the serving state (the
/// stacks are not touched between tokens).
#[test]
#[ignore = "the expert matvec bench: timing only, run on the device with --ignored"]
fn the_expert_matvec_bandwidth_bench() {
    if !testkit::device_present("the_expert_matvec_bandwidth_bench") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (warm_iters, cold_iters) = if cfg!(debug_assertions) {
        (8, 2)
    } else {
        (200, 9)
    };

    // The L2 flush: a dense Q8_0 matvec streaming 64k rows of k 1024 (68 MB).
    let (flush_k, flush_rows) = (1024_usize, 64_000_usize);
    let flush_w_bytes = QuantType::Q80
        .weight_bytes(flush_rows, flush_k)
        .expect("size");
    let (flush_w, flush_x, flush_y) = (
        testkit::upload_filled(&ctx, 0x5A, flush_w_bytes),
        testkit::upload_filled(&ctx, 0x5A, q8_1_bytes(flush_k, 1).expect("size")),
        testkit::upload_filled(&ctx, 0x5A, flush_rows * 4),
    );
    let flush = || {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: flush_w.range(0, flush_w_bytes).expect("flush w"),
                x_q8: flush_x
                    .range(0, q8_1_bytes(flush_k, 1).expect("size"))
                    .expect("flush x"),
                y: flush_y.range(0, flush_rows * 4).expect("flush y"),
                k: flush_k,
                rows: flush_rows,
                ncols: 1,
            },
        )
        .expect("flush matvec");
    };

    println!(
        "warm = {warm_iters} launches between two events; cold = one launch timed alone \
             right after the 68 MB flush (L2 evicted); GB/s counts the weight bytes read"
    );
    println!("shape                          warm_us  cold_us   GB/s_warm GB/s_cold");

    // The expert rows: (name, k, rows, x_per_slot). The stack rig uploads 16 experts, 8 used.
    let expert_shapes: [(&str, usize, usize, bool); 2] = [
        ("gate/up 2048x512 shared", 2048, 512, false),
        ("down 512x2048 per-slot", 512, 2048, true),
    ];
    for (name, k, rows, per_slot) in expert_shapes {
        let (stack, ids, shared, per_slot_x) = stack_rig(
            &ctx,
            &stream,
            QuantType::Q4K,
            (k, rows),
            0xB000_0000 + rows as u64,
        );
        let x = if per_slot { &per_slot_x } else { &shared };
        let x_bytes = q8_1_bytes(k, if per_slot { N_USED } else { 1 }).expect("size");
        let y_bytes = N_USED * rows * 4;
        let y = nan_filled(&ctx, y_bytes);
        let launch = || {
            ctx.matvec_expert(
                &stream,
                ExpertMatvecCall {
                    ty: QuantType::Q4K,
                    w_stack: stack
                        .range(
                            0,
                            QuantType::Q4K
                                .weight_bytes(rows * N_EXPERT, k)
                                .expect("size"),
                        )
                        .expect("stack"),
                    glu_gate: None,
                    ids: ids.range(0, N_USED * 4).expect("ids"),
                    x_q8: x.range(0, x_bytes).expect("x"),
                    x_per_slot: per_slot,
                    y: y.range(0, y_bytes).expect("y"),
                    y_slot_stride: rows,
                    k,
                    rows,
                    n_expert: N_EXPERT,
                    n_used: N_USED,
                },
            )
            .expect("expert matvec");
        };
        let warm_us = bench(&ctx, &stream, warm_iters, &launch);
        let mut cold = Vec::new();
        for _ in 0..cold_iters {
            flush();
            let start = ctx.bench_mark(&stream).expect("start");
            launch();
            let end = ctx.bench_mark(&stream).expect("end");
            stream.synchronize().expect("drain");
            cold.push(ctx.bench_elapsed_ms(start, end).expect("elapsed") * 1000.0);
            ctx.bench_destroy(start).expect("destroy");
            ctx.bench_destroy(end).expect("destroy");
        }
        cold.sort_by(|a, b| a.total_cmp(&b));
        let cold_us = cold[cold.len() / 2];
        let read_bytes =
            N_USED as f64 * rows as f64 * QuantType::Q4K.weight_bytes(1, k).expect("size") as f64;
        println!(
            "{name:30} {warm_us:8.1} {cold_us:8.1} {:9.0} {:9.0}",
            read_bytes / (warm_us as f64 * 1000.0),
            read_bytes / (cold_us as f64 * 1000.0)
        );
    }

    // The dense comparators: a contiguous Q4_K w of the SAME total bytes the expert launch
    // reads (8 x rows), so warm/cold and GB/s compare like for like.
    for (name, k, rows) in [
        ("dense Q4K 2048x4096", 2048_usize, 4096_usize),
        ("dense Q4K 512x16384", 512, 16384),
    ] {
        let mut rng = testkit::Rng::new(0xC000_0000 + rows as u64);
        let w_bytes = QuantType::Q4K.weight_bytes(rows, k).expect("size");
        let w = testkit::upload_bytes(
            &ctx,
            &reference::build_weight(QuantType::Q4K, rows, k, &mut rng),
        );
        let x: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let x_q8 = testkit::upload_filled(&ctx, 0xAA, q8_1_bytes(k, 1).expect("size"));
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x_dev.range(0, x.len() * 4).expect("x"),
                y: x_q8.range(0, q8_1_bytes(k, 1).expect("size")).expect("y"),
                k,
                ncols: 1,
            },
        )
        .expect("quantize");
        stream.synchronize().expect("drain");
        let y_bytes = rows * 4;
        let y = nan_filled(&ctx, y_bytes);
        let launch = || {
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty: QuantType::Q4K,
                    w: w.range(0, w_bytes).expect("w"),
                    x_q8: x_q8.range(0, q8_1_bytes(k, 1).expect("size")).expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols: 1,
                },
            )
            .expect("dense matvec");
        };
        let warm_us = bench(&ctx, &stream, warm_iters, &launch);
        let mut cold = Vec::new();
        for _ in 0..cold_iters {
            flush();
            let start = ctx.bench_mark(&stream).expect("start");
            launch();
            let end = ctx.bench_mark(&stream).expect("end");
            stream.synchronize().expect("drain");
            cold.push(ctx.bench_elapsed_ms(start, end).expect("elapsed") * 1000.0);
            ctx.bench_destroy(start).expect("destroy");
            ctx.bench_destroy(end).expect("destroy");
        }
        cold.sort_by(|a, b| a.total_cmp(&b));
        let cold_us = cold[cold.len() / 2];
        let read_bytes = w_bytes as f64;
        println!(
            "{name:30} {warm_us:8.1} {cold_us:8.1} {:9.0} {:9.0}",
            read_bytes / (warm_us as f64 * 1000.0),
            read_bytes / (cold_us as f64 * 1000.0)
        );
    }
}

/// `INFERENCE_CUDA_MVQ_WPR=1` (cc 7.2, default off) reassociates the Q4_K expert matvec's
/// reduction - one warp per row - so it cannot equal the dense launch's bits; the flag-on arm
/// must stay within a handful of ULPs of the dense oracle on every slot and both halves. The
/// flag is read per call, so the test arms it around its launches only (device tests run with
/// --test-threads=1, so the process-global env is safe here).
#[test]
fn matvec_expert_wpr_mode_matches_dense_within_reassociation() {
    const NAME: &str = "matvec_expert_wpr_mode_matches_dense_within_reassociation";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ty = QuantType::Q4K;
    inference_cuda_test_arm_mvq_wpr(1);
    for (si, shape) in [GATE_UP, DOWN, PADDED].iter().enumerate() {
        let (_, rows) = *shape;
        let stride = 2 * rows;
        let seed = 0x8200_0000 + si as u64;
        let (stack, ids, shared, per_slot) = stack_rig(&ctx, &stream, ty, *shape, seed);
        for (half, x, y_off, col) in [
            ("gate", &shared, 0usize, 0usize),
            ("up", &per_slot, rows * 4, 1),
        ] {
            let y = testkit::upload_filled(&ctx, 0xAA, stride * N_USED * 4);
            expert_launch(
                &ctx,
                &stream,
                ty,
                &stack,
                &ids,
                x,
                &y,
                y_off,
                *shape,
                half == "up",
                stride,
            );
            stream.synchronize().expect("drain");
            let got = testkit::read_bytes(&ctx, &y, stride * N_USED * 4);
            for (slot, id) in IDS.iter().enumerate() {
                // gate (shared x): every slot reads row 0; up (per-slot x): slot s reads row s.
                let want = dense_slot(
                    &ctx,
                    &stream,
                    ty,
                    &stack,
                    x,
                    *shape,
                    *id as usize,
                    if col == 0 { 0 } else { slot },
                );
                let got_f = &got[slot * stride * 4 + y_off..slot * stride * 4 + y_off + rows * 4];
                let mut worst = 0.0f32;
                let mut worst_ulps = 0u32;
                for (g, w) in got_f.chunks_exact(4).zip(want.chunks_exact(4)) {
                    let g = f32::from_le_bytes(g.try_into().expect("f32"));
                    let w = f32::from_le_bytes(w.try_into().expect("f32"));
                    assert!(
                        g.is_finite() && w.is_finite(),
                        "{NAME}: non-finite at {shape:?} slot {slot}"
                    );
                    // reassociation error scales with the term count and the cancellation, not
                    // the ULP of the result: the k 2048/512 shapes hold 0 ULPs, the padded
                    // k 2304 row legitimately drifts tens of ULPs, so the bound is relative.
                    let rel = (g - w).abs() / w.abs().max(1.0);
                    worst = worst.max(rel);
                    worst_ulps = worst_ulps.max(g.to_bits().abs_diff(w.to_bits()));
                    assert!(
                        rel <= 1e-4,
                        "{NAME}: {shape:?} {half} slot {slot} reassociated beyond 1e-4 rel ({rel})"
                    );
                }
                println!(
                    "{NAME}: {shape:?} {half} slot {slot} worst {worst:.2e} rel, {worst_ulps} ulps"
                );
            }
        }
    }
    inference_cuda_test_arm_mvq_wpr(0);
}
