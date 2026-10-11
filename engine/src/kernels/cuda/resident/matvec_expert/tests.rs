//! Device tests (3090) of `matvec_expert` (seam 1: the Rust wrapper against the C entry against
//! the V-MV1-trimmed vendored `mul_mat_vec_q` ids path). Device tests assume `--test-threads=1`
//! (decision 0040).
//!
//! The host reference is the one of `matvec_q`'s tests (`reference`), shared so the two files
//! cannot drift: the dense per-slot oracle of test 5 is `matvec_q` itself on the selected
//! expert's weights.

use super::{ExpertMatvecCall, ExpertShape, expert_needs};
use crate::EngError;
use crate::kernels::cuda::resident::matvec_q::tests::{Owner, assert_bad_request, reference};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceStream};

const N_EXPERT: usize = 16;
const N_USED: usize = 8;
/// Slots include 0 and 15 and a repeated id, per the packet.
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
/// shapes have k % 512 == 0, where the two strides coincide; KUBDAN-MSG-64 item 8, mutant S1).
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
        },
    )
    .expect("expert matvec");
}

/// The byte counts `plan_matvec_expert` checks a call against (KUBDAN-MSG-64 items 6 and 10, mutant
/// S4): the A2 footprint of `y`, the per-slot activation rows and the checked `rows * n_expert`.
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
    // Overflows are None, not a panic or a wrapped size: rows * n_expert (KUBDAN-MSG-64 F10), the
    // y footprint, and a shape without used slots.
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
    // S1 follow-up 1 (KUBDAN-MSG-67 condition 2): a `rows * n_expert` that WRAPS TO ZERO while
    // every later product stays small, so only the checked multiply refuses it. rows 1<<61 x 8
    // experts wraps to 0 (a wrapping form would answer weight_bytes(0, k) = 0 and pass), and
    // the y footprint (0 * stride + 1<<61, times 4 = 1<<63) still fits usize. The packet's
    // 1<<62 x 4 form wraps as well; BOTH forms are refused by the rows.checked_mul(n_expert)
    // multiply itself (1<<62 * 4 = 2^64 overflows it too — the second form's y footprint would
    // also overflow its own multiply, but the w_stack field is evaluated first); both are
    // pinned.
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
            // A2: y is [slot][gate|up]: one buffer of stride 2 * rows, the gate launch writes
            // offset 0 and the up launch offset rows of each slot.
            let (_, rows) = *shape;
            let stride = 2 * rows;
            // The full [slot][gate|up] buffer of A2: every slot has both halves, so the canary
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
            // The up launch at offset rows, and the per-slot activation view.
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

/// The A1 guard of one shape: ids holding -1 and `N_EXPERT` in two slots give those slots the quiet
/// NaN in every row and leave the other slots equal to the dense oracle.
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

/// Test 10, a real device test since A1: the gate/up shape (k 2048, one row per block) and the
/// production Q4_K down shape (k 512, 2048 rows, per-slot x): its K loop is shorter than one block
/// iteration, so the launch is the small-k one with several rows per block and the NaN loop of the
/// guard writes several rows there (KUBDAN-MSG-64 item 9, mutant S3: a loop that writes row 0 only).
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

/// Where `y` of an expert refusal case lies.
#[derive(Clone, Copy)]
enum YIn {
    Own,
    Stack,
    Ids,
    X,
}

/// One refusal case of the expert wrapper: the numbers of the call and the (offset, bytes) of its
/// ranges inside the rig's buffers.
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

/// Q4_K, k 768 (three blocks, padded to 1024), 16 experts of 64 rows, 8 used slots, the A2
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

/// The second adapter: a stream, a stack and an ids buffer that belong to another context.
struct OtherContext {
    stream: DeviceStream,
    stack: DeviceBuffer,
    ids: DeviceBuffer,
    _ctx: CudaContext,
}

/// A device buffer that holds `ids` as little-endian ints, zero padded to `total` bytes.
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
        },
    )
}

/// KUBDAN-MSG-64 item 6 for `plan_matvec_expert`: every refusal is a `BadRequest` with the text of
/// its own check (among them the A2 footprint of `y`), nothing is queued, the backend stays usable,
/// and the valid call of the same rig launches.
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
    // No refusal queued anything: y is untouched and the backend works; then the valid call runs.
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

/// A2 footprint check: the `rows` live f32 words of every slot differ from the poison word and
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
