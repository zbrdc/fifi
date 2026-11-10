//! The EXL3 MoE block of one real layer against an fp64 evaluation of the same pack.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory. The test loads the MoE tensors of one layer,
//! runs the block on a fixed normalized row and residual, and compares the router's picks, their
//! weights and the block output with a host evaluation: the router in fp64, every picked
//! expert's projections through the fp64 reference of the stored matrices, SiLU in fp64.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::{
    super::exl3_pack::{ExpertPart, expert_name, router_name, shared_gate_name, shared_name},
    common_tests::{dims, inventory},
    exl3_bank::{Exl3SlotResolver, ExpertStore},
    exl3_load_device_tests::{matrix_of, relative_l2, values_of},
    exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
    exl3_moe_source::{moe_weights, moe_weights_banked},
    exl3_source::PackReader,
    layer_table::LayerTable,
    moe::{Launch, MoeDims, RouteBufs},
    resources::STEP_ROWS,
};
use crate::{
    EngError,
    family::project::cuda::{CudaProjector, CudaWeights, Reserve},
    kernels::cuda::{DeviceBuffer, device_present, resident::router::router_logits_bytes},
    weights::exl3::{Checkpoint, Matrix},
};

const F32: usize = 4;

fn zeros(projector: &CudaProjector, bytes: usize) -> DeviceBuffer {
    projector.context().upload(&vec![0u8; bytes]).unwrap()
}

fn upload(projector: &CudaProjector, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    projector.context().upload(&bytes).unwrap()
}

fn read(projector: &CudaProjector, buffer: &DeviceBuffer) -> Vec<u8> {
    let stream = projector.context().stream().unwrap();
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(&stream, 0, &mut bytes).unwrap();
    bytes
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn project(matrix: &Matrix, input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.shape()[1]];
    matrix.project_fp64_into(input, 1, &mut out).unwrap();
    out
}

/// One expert (or the shared expert) on `x`: `down(silu(gate x) * up x)`.
fn expert(gate: &Matrix, up: &Matrix, down: &Matrix, x: &[f64]) -> Vec<f64> {
    let (gate, up) = (project(gate, x), project(up, x));
    let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    project(down, &act)
}

/// The router of the GGUF block in fp64: softmax over all logits, the top `top_k` by probability
/// then by lower index, weights renormalized over the picks, the sigmoid of the shared gate.
fn route(router: &[f32], gate: &[f32], x: &[f64], top_k: usize) -> (Vec<usize>, Vec<f64>, f64) {
    let dot = |row: &[f32]| -> f64 { row.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum() };
    let logits: Vec<f64> = router.chunks_exact(x.len()).map(dot).collect();
    let top = logits.iter().copied().fold(f64::MIN, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
    order.truncate(top_k);
    let picked: f64 = order.iter().map(|&e| weights[e] / total).sum();
    let picks = order.iter().map(|&e| weights[e] / total / picked).collect();
    (order, picks, 1.0 / (1.0 + (-dot(gate)).exp()))
}

fn inputs(hidden: usize) -> (Vec<f32>, Vec<f32>) {
    let row = (0..hidden)
        .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 700.0)
        .collect();
    let resid = (0..hidden)
        .map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0)
        .collect();
    (row, resid)
}

/// The router rows of the parity rounds: round 0 is the base row; every later round blends
/// the base row with its 337-rotation, taking the first blend whose host-routed top-k
/// shares three to five experts with the predecessor's. Rounds that share some - not all -
/// picks are what the banked overlap test needs: its resident group and its refilling
/// group both launch in one round. A positive scaling cannot produce this (the router is
/// linear, so scaling keeps the pick set), and the un-blended rotation replaces the whole
/// set (measured: disjoint top-8s). Sharing is a heuristic, not a hit guarantee (an
/// earlier miss can evict a shared pick's slot before the pick is processed); the test
/// asserts on the resolver's own stats, so a heuristic miss fails loudly, never vacuously.
fn rounded_rows(
    checkpoint: &Checkpoint,
    layer: usize,
    hidden: usize,
    top_k: usize,
) -> Vec<Vec<f32>> {
    let router = values_of(checkpoint, &router_name(layer));
    let gate = values_of(checkpoint, &shared_gate_name(layer));
    let (base, _) = inputs(hidden);
    let rotated: Vec<f32> = (0..hidden).map(|i| base[(i + 337) % hidden]).collect();
    let route_row = |row: &[f32]| -> Vec<usize> {
        let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        route(&router, &gate, &x, top_k).0
    };
    let mut rows = vec![base.clone()];
    let mut picks = route_row(&base);
    for _ in 1..6 {
        let mut blended = None;
        // The blend walk: at t=0 the picks equal the base set, at t=1 the rotated set;
        // rank crossings move members between the sets, so a step usually lands in the
        // 3..5 band (the fallback covers a walk that jumps over it).
        for step in 1..100u32 {
            let t = step as f32 / 100.0;
            let row: Vec<f32> = (0..hidden)
                .map(|i| base[i] * (1.0 - t) + rotated[i] * t)
                .collect();
            let p = route_row(&row);
            let shared = p.iter().filter(|e| picks.contains(e)).count();
            if (3..=5).contains(&shared) {
                blended = Some((row, p));
                break;
            }
        }
        let (row, p) = blended.unwrap_or_else(|| (rotated.clone(), route_row(&rotated)));
        rows.push(row);
        picks = p;
    }
    rows
}

struct Reference {
    ids: Vec<usize>,
    weights: Vec<f64>,
    out: Vec<f64>,
}

fn reference(checkpoint: &Checkpoint, layer: usize, moe: &MoeDims) -> Reference {
    let (row, resid) = inputs(moe.hidden);
    let x: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
    let router = values_of(checkpoint, &router_name(layer));
    let gate = values_of(checkpoint, &shared_gate_name(layer));
    let (ids, weights, shared_gate) = route(&router, &gate, &x, moe.top_k);
    let load = |name: String| matrix_of(checkpoint, &name);
    let mut out: Vec<f64> = resid.iter().map(|&v| f64::from(v)).collect();
    let mut delta = vec![0.0f64; moe.hidden];
    for (&e, weight) in ids.iter().zip(&weights) {
        let parts = ExpertPart::ALL.map(|part| load(expert_name(layer, e, part)));
        for (d, v) in delta
            .iter_mut()
            .zip(expert(&parts[0], &parts[1], &parts[2], &x))
        {
            *d += weight * v;
        }
    }
    let shared = ExpertPart::ALL.map(|part| load(shared_name(layer, part)));
    for (d, v) in delta
        .iter_mut()
        .zip(expert(&shared[0], &shared[1], &shared[2], &x))
    {
        *d += shared_gate * v;
    }
    out.iter_mut().zip(&delta).for_each(|(o, d)| *o += d);
    Reference { ids, weights, out }
}

/// Every range of one block on fresh zeroed buffers, owned by the caller.
struct Owned {
    buffers: Vec<DeviceBuffer>,
}

impl Owned {
    fn range<'a>(&'a self, index: usize) -> crate::kernels::cuda::DeviceRange<'a> {
        let buffer = &self.buffers[index];
        buffer.range(0, buffer.bytes()).unwrap()
    }
}

#[test]
#[ignore = "requires admitted GPU memory (4 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn real_pack_moe_block_matches_the_fp64_reference() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("real EXL3 moe block"), "a GPU is required");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).unwrap();
    let moe = MoeDims::new(&d, STEP_ROWS);
    let projector = CudaProjector::new().expect("projector");
    for layer in [3, 24] {
        let reader = PackReader::new(&projector, &checkpoint, Reserve::default());
        let loaded = moe_weights(&reader, &table, layer, &moe).expect("moe tensors");
        let weights = CudaWeights::from_exl3(loaded).unwrap();
        let block = Exl3MoeWeights::resolve(&weights, &table, layer, &moe).unwrap();
        let want = reference(&checkpoint, layer, &moe);
        let got = run_block(&projector, &block, &moe, false).expect("block");
        // The picks read on the device make the same block, byte for byte.
        let on_device = run_block(&projector, &block, &moe, true).expect("block, device picks");
        assert_eq!(
            on_device.ids, got.ids,
            "layer {layer}: device picks, the ids"
        );
        assert_eq!(
            on_device.out, got.out,
            "layer {layer}: device picks, the output"
        );
        assert_eq!(on_device.flag, 0, "layer {layer}: device picks, the flag");
        assert_eq!(got.ids, want.ids, "layer {layer}: the picks");
        for (g, w) in got.weights.iter().zip(&want.weights) {
            assert!(
                (f64::from(*g) - w).abs() < 1e-6 * w.max(1e-3),
                "layer {layer}: weight"
            );
        }
        let error = relative_l2(&got.out, &want.out);
        // The digest of the output bytes pins the block across a refactor: an unchanged path
        // prints the same digest.
        let bytes: Vec<u8> = got.out.iter().flat_map(|v| v.to_le_bytes()).collect();
        println!(
            "layer {layer}: picks {:?}, relative L2 {error:.3e}, output sha256 {:x}",
            got.ids,
            Sha256::digest(&bytes)
        );
        assert_eq!(got.flag, 0, "layer {layer}: arithmetic flag");
        assert!(error < 2e-5, "layer {layer}: {error}");
    }
}

struct Device {
    ids: Vec<usize>,
    weights: Vec<f32>,
    out: Vec<f32>,
    flag: u32,
}

fn run_block(
    projector: &CudaProjector,
    block: &Exl3MoeWeights<'_>,
    moe: &MoeDims,
    device_ids: bool,
) -> Result<Device, EngError> {
    let (row, _) = inputs(moe.hidden);
    run_block_rounded(projector, block, moe, device_ids, &row, None)
}

/// [`run_block`] with the given router row and the banked slot resolver; `bank` maps the
/// read picks through the layer's slot tables before the launches.
fn run_block_rounded(
    projector: &CudaProjector,
    block: &Exl3MoeWeights<'_>,
    moe: &MoeDims,
    device_ids: bool,
    row: &[f32],
    bank: Option<&mut crate::family::qwen36::cuda::exl3_bank::LayerResolver<'_, '_>>,
) -> Result<Device, EngError> {
    let (_, resid) = inputs(moe.hidden);
    let h = moe.hidden * F32;
    let logits = router_logits_bytes(crate::kernels::cuda::resident::router::RouterShape {
        m: 1,
        hidden: moe.hidden,
        n_expert: moe.n_expert,
        top_k: moe.top_k,
    })?;
    // Index: resid_in, resid_out, norm_row, logits, ids, weights, gate, status, gu, act, down,
    // shared_gu, shared_act, shared_out, xh, z, flag.
    let owned = Owned {
        buffers: vec![
            upload(projector, &resid),
            zeros(projector, h),
            upload(projector, row),
            zeros(projector, logits),
            zeros(projector, moe.top_k * 4),
            zeros(projector, moe.top_k * F32),
            zeros(projector, F32),
            zeros(projector, 4),
            zeros(projector, moe.top_k * 2 * moe.inter * F32),
            zeros(projector, moe.top_k * moe.inter * F32),
            zeros(projector, moe.top_k * h),
            zeros(projector, 2 * moe.shared_inter * F32),
            zeros(projector, moe.shared_inter * F32),
            zeros(projector, h),
            zeros(projector, 2 * moe.top_k * moe.hidden * F32),
            zeros(projector, 2 * moe.top_k * moe.inter.max(moe.hidden) * F32),
            zeros(projector, 4),
        ],
    };
    let r = |i| owned.range(i);
    let bufs = Exl3MoeBufs {
        route: RouteBufs {
            resid_in: r(0),
            resid_out: r(1),
            norm_row: r(2),
            logits: r(3),
            ids: r(4),
            weights: r(5),
            shared_gate: r(6),
            status: r(7),
        },
        expert_gu: r(8),
        expert_act: r(9),
        expert_down: r(10),
        shared_gu: r(11),
        shared_act: r(12),
        shared_out: r(13),
        scratch: Scratch {
            xh: r(14),
            z: r(15),
            flag: r(16),
        },
    };
    let ctx = projector.context();
    let stream = ctx.stream()?;
    // The overlap fact travels with the resolver (the arming of `enable_overlap`), so the
    // harness cannot desync from the production read of the split.
    let overlap = bank.as_ref().is_some_and(|bank| bank.overlap_armed());
    run_exl3_moe(
        &Launch {
            ctx,
            stream: &stream,
        },
        block,
        &bufs,
        moe,
        device_ids,
        bank,
        overlap,
    )?;
    stream.synchronize()?;
    let ids = read(projector, &owned.buffers[4])
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b) as usize)
        .collect();
    let flag = u32::from_le_bytes(read(projector, &owned.buffers[16])[..4].try_into().unwrap());
    Ok(Device {
        ids,
        weights: floats(&read(projector, &owned.buffers[5])),
        out: floats(&read(projector, &owned.buffers[1])),
        flag,
    })
}

/// The banked block against the resident block, byte for byte, over several input rounds.
///
/// The banked load keeps the SAME layer in an 8-slot bank (8 = the picks per token, the
/// admission floor) with every expert in the host store; each round scales the router row, so
/// the picks walk the experts and every round after the first misses and evicts. The banked
/// launches must produce the exact bytes of the resident block — the slot fill is a plain byte
/// move of the same staged layout, so nothing may differ by even a rounding step.
#[test]
#[ignore = "requires the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT and a GPU"]
fn banked_moe_block_is_byte_identical_to_the_resident_block() {
    banked_block_parity(false);
}

/// [`banked_moe_block_is_byte_identical_to_the_resident_block`] with the overlap armed
/// (`Q36_CUDA_BANK_OVERLAP`'s device path): the misses refill on the copy stream and the
/// resident picks' gate/up launches run before the event wait — the same bytes through a
/// different stream order is the whole claim of the split.
#[test]
#[ignore = "requires the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT and a GPU"]
fn overlapped_banked_moe_block_is_byte_identical_to_the_resident_block() {
    banked_block_parity(true);
}

fn banked_block_parity(overlap: bool) {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("banked EXL3 moe block"), "a GPU is required");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).unwrap();
    let moe = MoeDims::new(&d, STEP_ROWS);
    let projector = CudaProjector::new().expect("projector");
    const SLOTS: usize = 8;
    for layer in [3, 24] {
        let reader = PackReader::new(&projector, &checkpoint, Reserve::default());
        let loaded = moe_weights(&reader, &table, layer, &moe).expect("moe tensors");
        let weights = CudaWeights::from_exl3(loaded).unwrap();
        let block = Exl3MoeWeights::resolve(&weights, &table, layer, &moe).unwrap();
        let mut store = ExpertStore::with_layers(1);
        let banked_loaded = moe_weights_banked(&reader, &table, layer, &moe, SLOTS, &mut store)
            .expect("banked moe tensors");
        let banked_weights = CudaWeights::from_exl3(banked_loaded).unwrap();
        let banked_block =
            Exl3MoeWeights::resolve_banked(&banked_weights, &table, layer, &moe, SLOTS).unwrap();
        let mut resolver =
            Exl3SlotResolver::new(store, std::slice::from_ref(&banked_block), moe.n_expert, 0)
                .unwrap();
        if overlap {
            resolver.enable_overlap(projector.context()).unwrap();
        }
        let rows = rounded_rows(&checkpoint, layer, moe.hidden, moe.top_k);
        let (mut saw_miss, mut saw_mixed) = (false, false);
        for (round, row) in rows.iter().enumerate() {
            let want = run_block_rounded(&projector, &block, &moe, false, row, None)
                .expect("the resident block");
            let before = resolver.stats();
            let got = {
                let mut bank = resolver.layer(0);
                run_block_rounded(&projector, &banked_block, &moe, false, row, Some(&mut bank))
                    .expect("the banked block")
            };
            assert_eq!(got.ids, want.ids, "layer {layer} round {round}: the picks");
            assert_eq!(
                got.out, want.out,
                "layer {layer} round {round}: the output bytes"
            );
            assert_eq!(got.flag, 0, "layer {layer} round {round}: arithmetic flag");
            let (hits, misses) = resolver.stats();
            let (round_hits, round_misses) = (hits - before.0, misses - before.1);
            saw_miss |= round_misses > 0;
            saw_mixed |= round_hits > 0 && round_misses > 0;
            println!(
                "layer {layer} round {round}: picks {:?}, hits {round_hits}, misses {round_misses}",
                got.ids
            );
        }
        assert!(
            saw_miss,
            "layer {layer}: the rounds never missed; the resolver was not exercised"
        );
        assert!(
            saw_mixed,
            "layer {layer}: no round mixed hits and misses; the resident-before-wait \
             split of the overlapped block was never exercised"
        );
        let (hits, misses) = resolver.stats();
        assert_eq!(
            hits + misses,
            (rows.len() * moe.top_k) as u64,
            "every pick counted"
        );
    }
}
