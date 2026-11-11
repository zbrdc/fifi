//! The MoE block of a K-row verify window against K
//! single-row blocks, byte for byte, on synthetic EXL3 banks of the real Qwen3.6 shapes (hidden
//! 2048, experts 2048 x 512 and 512 x 2048, a shared expert of the same shapes; 32 experts instead
//! of 256 and the same top 8, so the bank is small).
//!
//! The K-row body of the verifier runs the router for all K rows at once (4-row chunks from 8
//! rows), then one chain of id-indexed launches per row (the routed experts read different weights
//! per row, so one launch cannot serve them all), the shared expert for all K rows at once in the
//! interleaved layout (gate and up as 2K one-row groups), and one combine. The test builds
//! exactly that from the pieces:
//!
//! * `router_rows`, then `routed_on_device` called once per row with row-offset ranges of the K-row
//!   buffers (`ids`, `expert_gu`, `expert_act`, `expert_down` offset per row so that every row's
//!   results survive to the combine), then `shared_rows` and one K-row `moe_combine`;
//! * against `run_exl3_moe` on one row at a time (the block the single-row step runs), on separate
//!   buffers.
//!
//! Every intermediate (ids, pick weights, shared gate, the routed `down` rows, the shared expert's
//! output) and the block output are compared, so a failure names the stage. Mutants killed: a row
//! whose chain uses row 0's ids, all rows sharing one `expert_down` or `ids` slice, a shared expert
//! laid out as two K-row groups, and any row-offset mistake of the K-row body. A host without a
//! device skips them.

use std::collections::HashMap;

use half::f16;

use super::{
    exl3_moe::{
        Exl3MoeBufs, Exl3MoeWeights, GROUPED_PAIRS, Scratch, SharedRowsIo, SharedWeights,
        run_exl3_moe, run_exl3_moe_rows, shared, shared_rows,
    },
    moe::{Launch, MoeDims, RouteBufs},
};
use crate::{
    EngError,
    family::project::cuda::{CudaProjector, Reserve, exl3::Exl3Bank},
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Config, Matrix},
    },
};

const F32: usize = 4;
const HIDDEN: usize = 2048;
const INTER: usize = 512;
const EXPERTS: usize = 32;
const TOP_K: usize = 8;
/// 16 x 16 tiles carry `bits_twice * 8` words: 8 is 4 bits per weight.
const BITS_TWICE: usize = 8;
const MUL1_MARKER: u32 = 0x83dc_d12d;

struct Components(HashMap<String, Tensor>);

impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

fn pseudo(seed: u32, count: usize) -> Vec<u16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u16
        })
        .collect()
}

fn half_vector(seed: u32, count: usize) -> Tensor {
    let bytes = pseudo(seed, count)
        .into_iter()
        .flat_map(|word| f16::from_f32((f32::from(word % 255) + 1.0) / 256.0).to_le_bytes())
        .collect();
    Tensor::try_from_host(Dtype::F16, vec![count as u32], bytes).unwrap()
}

/// A matrix of `shape` with its own trellis and scales per `seed`, in the `mul1` codebook.
fn member(seed: u32, shape: [usize; 2]) -> Matrix {
    let [k, n] = shape;
    let words = BITS_TWICE * 8;
    let trellis: Vec<u8> = pseudo(seed, k / 16 * (n / 16) * words)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let parts = HashMap::from([
        (
            "m.trellis".to_string(),
            Tensor::try_from_host(
                Dtype::I16,
                vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                trellis,
            )
            .unwrap(),
        ),
        ("m.suh".to_string(), half_vector(seed + 1, k)),
        ("m.svh".to_string(), half_vector(seed + 2, n)),
        (
            "m.mul1".to_string(),
            Tensor::try_from_host(Dtype::I32, vec![], MUL1_MARKER.to_le_bytes().to_vec()).unwrap(),
        ),
    ]);
    let config = Config::from_json(
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.4.9","codebook":"mul1"}}"#
            .as_slice(),
    )
    .unwrap();
    Matrix::load(&Components(parts), "m", &config).unwrap()
}

/// A small xorshift64* generator with a Box-Muller normal (`kernels::cuda::testkit` is private to
/// the kernel tree).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

fn normal(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

fn upload(ctx: &CudaContext, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    ctx.upload(&bytes).unwrap()
}

fn zeros(ctx: &CudaContext, bytes: usize) -> DeviceBuffer {
    ctx.upload(&vec![0u8; bytes]).unwrap()
}

fn whole(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).unwrap()
}

fn rows_of<'b>(
    buffer: &'b DeviceBuffer,
    row_bytes: usize,
    first: usize,
    rows: usize,
) -> DeviceRange<'b> {
    buffer.range(first * row_bytes, rows * row_bytes).unwrap()
}

fn read(ctx: &CudaContext, stream: &DeviceStream, buffer: &DeviceBuffer) -> Vec<u8> {
    stream.synchronize().expect("synchronize");
    let mut bytes = vec![0; buffer.bytes()];
    ctx.download(buffer, &mut bytes).expect("download");
    bytes
}

struct Weights {
    router: DeviceBuffer,
    router_shared: DeviceBuffer,
    gate: Exl3Bank,
    up: Exl3Bank,
    down: Exl3Bank,
    shared_gate: crate::family::project::cuda::exl3::Exl3Weight,
    shared_up: crate::family::project::cuda::exl3::Exl3Weight,
    shared_down: crate::family::project::cuda::exl3::Exl3Weight,
}

fn build_weights(projector: &CudaProjector) -> Weights {
    let ctx = projector.context();
    let mut rng = Rng::new(0x4b52_0090);
    let reserve = Reserve::default();
    let bank = |seed: u32, shape: [usize; 2]| {
        projector
            .upload_exl3_bank(EXPERTS, reserve, |i| {
                Ok(member(seed + 10 * i as u32, shape))
            })
            .expect("bank")
    };
    let single = |seed: u32, shape: [usize; 2]| {
        projector
            .upload_exl3(&member(seed, shape), reserve)
            .expect("shared matrix")
    };
    Weights {
        router: upload(ctx, &normal(&mut rng, EXPERTS * HIDDEN, 0.05)),
        router_shared: upload(ctx, &normal(&mut rng, HIDDEN, 0.05)),
        gate: bank(1_000, [HIDDEN, INTER]),
        up: bank(20_000, [HIDDEN, INTER]),
        down: bank(40_000, [INTER, HIDDEN]),
        shared_gate: single(60_000, [HIDDEN, INTER]),
        shared_up: single(61_000, [HIDDEN, INTER]),
        shared_down: single(62_000, [INTER, HIDDEN]),
    }
}

fn dims(rows: usize) -> MoeDims {
    MoeDims {
        rows,
        hidden: HIDDEN,
        n_expert: EXPERTS,
        top_k: TOP_K,
        inter: INTER,
        shared_inter: INTER,
    }
}

/// The ranges of `rows` rows of the block's scratch and outputs, plus the projections' shared scratch.
struct Buffers {
    resid_in: DeviceBuffer,
    resid_out: DeviceBuffer,
    norm_row: DeviceBuffer,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    gate: DeviceBuffer,
    status: DeviceBuffer,
    expert_gu: DeviceBuffer,
    expert_act: DeviceBuffer,
    expert_down: DeviceBuffer,
    shared_gu: DeviceBuffer,
    shared_act: DeviceBuffer,
    shared_out: DeviceBuffer,
    xh: DeviceBuffer,
    z: DeviceBuffer,
    flag: DeviceBuffer,
}

fn buffers(ctx: &CudaContext, rows: usize, resid: &[f32], norm: &[f32]) -> Buffers {
    let logits = router_logits_bytes(RouterShape {
        m: rows,
        hidden: HIDDEN,
        n_expert: EXPERTS,
        top_k: TOP_K,
    })
    .unwrap();
    Buffers {
        resid_in: upload(ctx, resid),
        resid_out: zeros(ctx, rows * HIDDEN * F32),
        norm_row: upload(ctx, norm),
        logits: zeros(ctx, logits),
        ids: zeros(ctx, rows * TOP_K * 4),
        weights: zeros(ctx, rows * TOP_K * F32),
        gate: zeros(ctx, rows * F32),
        status: zeros(ctx, 4),
        expert_gu: zeros(ctx, rows * TOP_K * 2 * INTER * F32),
        expert_act: zeros(ctx, rows * TOP_K * INTER * F32),
        expert_down: zeros(ctx, rows * TOP_K * HIDDEN * F32),
        shared_gu: zeros(ctx, rows * 2 * INTER * F32),
        shared_act: zeros(ctx, rows * INTER * F32),
        shared_out: zeros(ctx, rows * HIDDEN * F32),
        // The projections' scratch is reused by every launch in stream order: the indexed
        // launch's 16 pairs, the shared gate/up's pair batches (up to GROUPED_PAIRS rows), and
        // the down windows (up to `rows` rows of xh and z).
        xh: zeros(ctx, GROUPED_PAIRS.max(rows) * HIDDEN * F32),
        z: zeros(ctx, GROUPED_PAIRS.max(rows) * HIDDEN * F32),
        flag: zeros(ctx, 4),
    }
}

impl Buffers {
    fn scratch(&self) -> Scratch<'_> {
        Scratch {
            xh: whole(&self.xh),
            z: whole(&self.z),
            flag: whole(&self.flag),
        }
    }

    /// Row `r` of `rows` as the one-row block's ranges: every per-row buffer offset by `r` rows.
    fn row(&self, r: usize) -> Exl3MoeBufs<'_> {
        Exl3MoeBufs {
            route: RouteBufs {
                resid_in: rows_of(&self.resid_in, HIDDEN * F32, r, 1),
                resid_out: rows_of(&self.resid_out, HIDDEN * F32, r, 1),
                norm_row: rows_of(&self.norm_row, HIDDEN * F32, r, 1),
                logits: rows_of(&self.logits, (EXPERTS + 1) * F32, r, 1),
                ids: rows_of(&self.ids, TOP_K * 4, r, 1),
                weights: rows_of(&self.weights, TOP_K * F32, r, 1),
                shared_gate: rows_of(&self.gate, F32, r, 1),
                status: whole(&self.status),
            },
            expert_gu: rows_of(&self.expert_gu, TOP_K * 2 * INTER * F32, r, 1),
            expert_act: rows_of(&self.expert_act, TOP_K * INTER * F32, r, 1),
            expert_down: rows_of(&self.expert_down, TOP_K * HIDDEN * F32, r, 1),
            shared_gu: rows_of(&self.shared_gu, 2 * INTER * F32, r, 1),
            shared_act: rows_of(&self.shared_act, INTER * F32, r, 1),
            shared_out: rows_of(&self.shared_out, HIDDEN * F32, r, 1),
            scratch: self.scratch(),
        }
    }
}

impl Buffers {
    /// The first `rows` rows of every buffer, as the K-row block takes them.
    fn rows(&self, rows: usize) -> Exl3MoeBufs<'_> {
        Exl3MoeBufs {
            route: RouteBufs {
                resid_in: rows_of(&self.resid_in, HIDDEN * F32, 0, rows),
                resid_out: rows_of(&self.resid_out, HIDDEN * F32, 0, rows),
                norm_row: rows_of(&self.norm_row, HIDDEN * F32, 0, rows),
                logits: rows_of(&self.logits, (EXPERTS + 1) * F32, 0, rows),
                ids: rows_of(&self.ids, TOP_K * 4, 0, rows),
                weights: rows_of(&self.weights, TOP_K * F32, 0, rows),
                shared_gate: rows_of(&self.gate, F32, 0, rows),
                status: whole(&self.status),
            },
            expert_gu: rows_of(&self.expert_gu, TOP_K * 2 * INTER * F32, 0, rows),
            expert_act: rows_of(&self.expert_act, TOP_K * INTER * F32, 0, rows),
            expert_down: rows_of(&self.expert_down, TOP_K * HIDDEN * F32, 0, rows),
            shared_gu: rows_of(&self.shared_gu, 2 * INTER * F32, 0, rows),
            shared_act: rows_of(&self.shared_act, INTER * F32, 0, rows),
            shared_out: rows_of(&self.shared_out, HIDDEN * F32, 0, rows),
            scratch: self.scratch(),
        }
    }
}

fn block<'w>(w: &'w Weights) -> Exl3MoeWeights<'w> {
    Exl3MoeWeights {
        router: &w.router,
        router_shared: &w.router_shared,
        gate: &w.gate,
        up: &w.up,
        down: &w.down,
        shared_gate: &w.shared_gate,
        shared_up: &w.shared_up,
        shared_down: &w.shared_down,
    }
}

/// The K-row block of a verify window, the function the step runs: the router for the K rows, one
/// routed chain per row on row-offset ranges, the shared expert for the K rows, one combine.
fn run_rows(launch: &Launch<'_>, w: &Weights, b: &Buffers, rows: usize) -> Result<(), EngError> {
    run_exl3_moe_rows(launch, &block(w), &b.rows(rows), &dims(rows))
}

/// One stage of a block, as the bytes of its buffers.
fn stages(b: &Buffers) -> [(&'static str, &DeviceBuffer); 6] {
    [
        ("ids", &b.ids),
        ("pick weights", &b.weights),
        ("shared gate", &b.gate),
        ("routed down rows", &b.expert_down),
        ("shared expert output", &b.shared_out),
        ("block output", &b.resid_out),
    ]
}

/// A K-row window of the MoE block equals K single-row blocks, stage by
/// stage, for K = 2 to 8.
#[test]
fn the_moe_block_of_k_rows_equals_k_single_row_blocks() {
    if !device_present("K-row MoE block against K single-row blocks") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let w = build_weights(&projector);
    let mut rng = Rng::new(0x4b52_0091);
    let weights = block(&w);
    for rows in 2..=8usize {
        let resid = normal(&mut rng, rows * HIDDEN, 1.0);
        let norm = normal(&mut rng, rows * HIDDEN, 1.0);
        let batch = buffers(ctx, rows, &resid, &norm);
        run_rows(&launch, &w, &batch, rows).expect("the K-row body");
        let batch_bytes: Vec<Vec<u8>> = stages(&batch)
            .iter()
            .map(|(_, buffer)| read(ctx, &stream, buffer))
            .collect();
        // The picks must differ across the rows, or a shared-ids mutant would not show.
        let ids: Vec<i32> = batch_bytes[0]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        assert!(
            ids.chunks(TOP_K).any(|row| row != &ids[..TOP_K]),
            "K {rows}: every row picked the same experts; the test data cannot tell rows apart"
        );
        for r in 0..rows {
            let row_resid = &resid[r * HIDDEN..(r + 1) * HIDDEN];
            let row_norm = &norm[r * HIDDEN..(r + 1) * HIDDEN];
            let single = buffers(ctx, 1, row_resid, row_norm);
            run_exl3_moe(
                &launch,
                &weights,
                &single.row(0),
                &dims(1),
                true,
                None,
                false,
            )
            .expect("one-row block");
            for (i, (name, buffer)) in stages(&single).iter().enumerate() {
                let want = read(ctx, &stream, buffer);
                let width = want.len();
                assert_eq!(
                    &batch_bytes[i][r * width..(r + 1) * width],
                    &want[..],
                    "K {rows} row {r}: {name}"
                );
            }
        }
    }
}

/// The shared expert with gate and up as `2K` one-row groups in one launch, one SwiGLU at
/// `ld_in = 2 s`, one K-row down group) equals K runs of the one-row `shared()` block, and the
/// launch is accepted unchanged (the SwiGLU's shape check is not relaxed).
#[test]
fn the_shared_expert_of_k_rows_equals_k_one_row_blocks() {
    if !device_present("K-row shared expert against K one-row blocks") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let w = build_weights(&projector);
    let weights = block(&w);
    let mut rng = Rng::new(0x4b52_0092);
    // 2..=8 pins the single-batch widths (one gate/up batch, one down group); 9 crosses the
    // down-group bound; 25/33/49 cross the 24-row gate/up pair bound into two and three
    // batches; 64 is the widest planned window (three batches).
    for rows in [2usize, 3, 8, 9, 25, 33, 49, 64] {
        let norm = normal(&mut rng, rows * HIDDEN, 1.0);
        let batch = buffers(ctx, rows, &vec![0.0; rows * HIDDEN], &norm);
        shared_rows(
            &launch,
            &SharedWeights {
                gate: &w.shared_gate,
                up: &w.shared_up,
                down: &w.shared_down,
            },
            &SharedRowsIo {
                norm_row: whole(&batch.norm_row),
                gu: whole(&batch.shared_gu),
                act: whole(&batch.shared_act),
                out: whole(&batch.shared_out),
                scratch: batch.scratch(),
            },
            (HIDDEN, INTER, rows),
        )
        .expect("shared_rows");
        let got = [
            read(ctx, &stream, &batch.shared_gu),
            read(ctx, &stream, &batch.shared_act),
            read(ctx, &stream, &batch.shared_out),
        ];
        for r in 0..rows {
            let single = buffers(
                ctx,
                1,
                &vec![0.0; HIDDEN],
                &norm[r * HIDDEN..(r + 1) * HIDDEN],
            );
            shared(&launch, &weights, &single.row(0), &dims(1)).expect("one-row shared");
            let want = [
                read(ctx, &stream, &single.shared_gu),
                read(ctx, &stream, &single.shared_act),
                read(ctx, &stream, &single.shared_out),
            ];
            for (i, name) in ["gate|up rows", "SwiGLU rows", "down rows"]
                .iter()
                .enumerate()
            {
                let width = want[i].len();
                assert_eq!(
                    &got[i][r * width..(r + 1) * width],
                    &want[i][..],
                    "K {rows} row {r}: {name}"
                );
            }
        }
    }
}

/// The K-row shared expert refuses one row and windows over `MAX_EXL3_ROWS` (64): the one-row
/// block is `shared()`, and both families' planners cap at the shared bound.
#[test]
fn the_k_row_shared_expert_refuses_one_row_and_windows_over_sixty_four() {
    if !device_present("K-row shared expert row counts") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let w = build_weights(&projector);
    let b = buffers(ctx, 9, &vec![0.0; 9 * HIDDEN], &vec![0.0; 9 * HIDDEN]);
    for (rows, wanted) in [
        (0usize, "at least 2 rows"),
        (1, "at least 2 rows"),
        (65, "at most 64 rows"),
    ] {
        let result = shared_rows(
            &launch,
            &SharedWeights {
                gate: &w.shared_gate,
                up: &w.shared_up,
                down: &w.shared_down,
            },
            &SharedRowsIo {
                norm_row: whole(&b.norm_row),
                gu: whole(&b.shared_gu),
                act: whole(&b.shared_act),
                out: whole(&b.shared_out),
                scratch: b.scratch(),
            },
            (HIDDEN, INTER, rows),
        );
        match result {
            Err(EngError::BadRequest(message)) if message.contains(wanted) => {}
            other => panic!("rows {rows}: expected the {wanted:?} refusal, got {other:?}"),
        }
    }
}

/// FIFI_EXL3_GROUP_SPLIT 4/8 slice k further than the byte-pinned 2: more blocks in
/// flight for the fill-limited decode, at a different but fixed partial fold order.
/// The routed block stays within a small tolerance of the split-2 output (float reorder
/// of the same sums, not a different computation), and the override returns to default.
#[test]
fn split_4_and_8_stay_within_reorder_tolerance_of_split_2() {
    if !device_present("grouped split 4/8 tolerance") {
        return;
    }
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let w = build_weights(&projector);
    let mut rng = Rng::new(0x9d5c_2e11);
    let resid = normal(&mut rng, HIDDEN, 1.0);
    let norm = normal(&mut rng, HIDDEN, 1.0);
    let mut outs = Vec::with_capacity(3);
    for split in [2u32, 4, 8] {
        super::exl3_moe::set_group_split_override(split);
        let mut b = buffers(ctx, 1, &resid, &norm);
        // Seven partial banks (split 8) of the widest n the block launches.
        b.z = zeros(ctx, 20 * HIDDEN * F32 * 7);
        run_exl3_moe(&launch, &block(&w), &b.row(0), &dims(1), true, None, false)
            .expect("the one-row block");
        let bytes = read(ctx, &stream, &b.resid_out);
        outs.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect::<Vec<_>>(),
        );
    }
    super::exl3_moe::set_group_split_override(0);
    let scale = outs[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (split, got) in [(4u32, &outs[1]), (8, &outs[2])] {
        // Reordering the k fold changes float rounding only: the drift stays at the
        // eps*sqrt(k)*scale level (k = 2048 here). A wrong sum would be O(scale).
        let worst = got
            .iter()
            .zip(&outs[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-5 * scale,
            "split {split}: worst |delta| {worst} at scale {scale} over {} outputs",
            got.len()
        );
    }
}
