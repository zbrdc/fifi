//! Seam-1 determinism of the expert shapes of one Qwen3.6-35B-A3B decode step (decision 0047
//! item 7, packet RD-2e): `pack` once, then `matmul_expert` for the 8 selected slots, at m = 1,
//! 1,000 repeats per shape, every repeat byte-identical to the first. The weights are synthetic
//! blocks of the real type and shape (`testkit::upload_weight`); the activations are normal
//! values, not the small integers of `testkit::input`, which make every fp32 sum exact.
//!
//! The tests assume `--test-threads=1`.

use super::{
    CudaContext, ExpertCall, ExpertSelect, ExpertStack, GgufType, MatmulIo, PackCall, ids_buffer,
    native, testkit,
};
use crate::kernels::cuda::testkit::{Mismatch, REPEATS};

/// Experts of the Qwen3.6-35B-A3B stacks and the slots a token selects.
const EXPERTS: usize = 256;
const SLOTS: usize = 8;
/// Eight distinct experts, including the first and the last.
const IDS: [i32; SLOTS] = [0, 255, 17, 128, 64, 200, 3, 99];

/// One expert stack: `rows` per expert x `k`, Q4_K.
struct Shape {
    name: &'static str,
    rows: usize,
    k: usize,
}

/// E1 (`ffn_gate_exps`, `ffn_up_exps`) and E2 (`ffn_down_exps`) of the packet.
const SHAPES: [Shape; 2] = [
    Shape {
        name: "expert_gate_up",
        rows: 512,
        k: 2048,
    },
    Shape {
        name: "expert_down",
        rows: 2048,
        k: 512,
    },
];

fn repeat_expert(ctx: &CudaContext, index: usize, shape: &Shape) {
    let Shape { name, rows, k } = *shape;
    let weight = testkit::upload_weight(ctx, GgufType::Q4K, EXPERTS * rows, k);
    let mut rng = testkit::Rng::new(0x52D2_E100 + index as u64);
    let values: Vec<f32> = (0..k).map(|_| rng.normal() as f32).collect();
    let x = testkit::upload_f32(ctx, &values);
    let ids = ids_buffer(ctx, &IDS, 256);
    let need = native::activation_bytes(rows, k, 1).expect("sizes of the call");
    assert_eq!(need.y, rows * 4, "{name}: y holds one f32 per row");
    let xf = ctx.allocate(need.xf).expect("xf");
    let stream = ctx.stream().expect("stream");
    let each = need.y;
    let out_bytes = SLOTS * each;
    let want = testkit::run_repeats(ctx, &stream, out_bytes, |_, y| {
        let pack = PackCall {
            x: x.range(0, need.x).expect("x range"),
            xf: xf.range(0, need.xf).expect("xf range"),
            m: 1,
            k,
        };
        ctx.pack(&stream, pack).expect("queue the pack");
        for slot in 0..SLOTS {
            let call = ExpertCall {
                stack: ExpertStack {
                    weight: &weight,
                    experts: EXPERTS,
                },
                select: ExpertSelect {
                    ids: ids.range(0, 32).expect("ids range"),
                    slot,
                    count: EXPERTS,
                },
                io: MatmulIo {
                    xf: xf.range(0, need.xf).expect("xf range"),
                    y: y.buffer()
                        .range(y.offset() + slot * each, each)
                        .expect("slot range"),
                    m: 1,
                },
            };
            ctx.matmul_expert(&stream, call).expect("queue the expert");
        }
    })
    .unwrap_or_else(|Mismatch { repeat, index }| {
        panic!(
            "{name}: repeat {repeat} differs from repeat 0 at byte {index} (element {})",
            index / 4
        )
    });
    assert_eq!(want.len(), out_bytes);
    println!(
        "projection_repeat: {name} Q4K {rows}x{k} repeats={REPEATS} y_bytes={out_bytes} slots={SLOTS}"
    );
}

#[test]
fn projection_repeat_expert_shapes_are_bitwise_equal() {
    let test = "projection_repeat_expert_shapes_are_bitwise_equal";
    if !testkit::device_present(test) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    for (index, shape) in SHAPES.iter().enumerate() {
        repeat_expert(&ctx, index, shape);
    }
}
