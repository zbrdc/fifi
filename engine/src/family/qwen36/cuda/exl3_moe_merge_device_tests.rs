//! Device tests of the shared expert inside the routed launches (`family/exl3/moe/merge.rs`). All are
//! `#[ignore]`: they need the 3090 held exclusively and the pinned 4.00 pack in
//! `FIFI_EXL3_CHECKPOINT`, and run in a window with its own authorization.
//!
//! Every comparison is on BYTES, never on a tolerance. The reference is `run_exl3_moe`; the
//! candidate is `run_exl3_moe_merged`. On the first difference a test stops and prints the first
//! differing 32-bit word with both values.
//!
//! - `shared_in_routed_block_bytes_equal_the_reference_block`: on two real layers and several
//!   routing patterns (different normalized rows give different picks), host picks and device
//!   picks: the ids, the pick weights, the routed gate/up rows, the shared gate/up, both swiglu
//!   outputs, the routed down outputs, the shared expert output, the combine output and the
//!   flag, and the zero word stays zero.
//! - `shared_in_routed_special_values_equal_the_reference`: rows of zeros, signed zeros,
//!   denormals, huge values that overflow in the rotation, and a non-finite value, with device
//!   picks: the same bytes and the same flag, including the pairs a faulty router skips.
//! - `shared_in_routed_pair_equals_the_grouped_pair_on_real_tensors`: the kernel level. The
//!   shared gate and up of a real layer through the 2-pair grouped launch of today and through
//!   rows 16 and 17 of the 18-pair indexed launch: the rotated inputs, the partials and the
//!   outputs; and the routed pairs of that launch against the default 16-pair launch.
//! - `shared_in_routed_graph_has_seven_fewer_kernel_nodes`: the captured block has exactly 7
//!   nodes fewer per layer.

use std::{collections::BTreeSet, path::Path};

use super::{
    common_tests::{dims, inventory},
    exl3_moe::{
        Exl3MoeBufs, Exl3MoeWeights, Scratch,
        merge::{SharedMerge, run_exl3_moe_merged},
        run_exl3_moe,
    },
    exl3_moe_source::moe_weights,
    exl3_source::PackReader,
    layer_table::LayerTable,
    moe::{Launch, MoeDims, RouteBufs},
    resources::STEP_ROWS,
};
use crate::{
    EngError,
    family::project::cuda::{
        CudaProjector, CudaWeights, Reserve,
        exl3::{Exl3Io, Exl3Weight},
    },
    kernels::cuda::{
        CapturePlan, DeviceBuffer, device_present,
        resident::{exl3::SingleRow, router::router_logits_bytes},
    },
    weights::exl3::Checkpoint,
};

const F32: usize = 4;
/// Layers of the pack to test: a DeltaNet-adjacent early layer and a late one.
const LAYERS: [usize; 2] = [3, 24];

fn upload(projector: &CudaProjector, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    projector.context().upload(&bytes).unwrap()
}

fn zeros(projector: &CudaProjector, bytes: usize) -> DeviceBuffer {
    projector.context().upload(&vec![0u8; bytes]).unwrap()
}

fn read(projector: &CudaProjector, buffer: &DeviceBuffer, offset: usize, bytes: usize) -> Vec<u8> {
    let stream = projector.context().stream().unwrap();
    let mut out = vec![0; bytes];
    buffer.read_at(&stream, offset, &mut out).unwrap();
    out
}

/// The first differing 32-bit word of two byte strings, as a message; `None` when equal.
fn full(b: &DeviceBuffer) -> crate::kernels::cuda::DeviceRange<'_> {
    b.range(0, b.bytes()).unwrap()
}

fn first_difference(name: &str, left: &[u8], right: &[u8]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!(
            "{name}: lengths differ ({} and {})",
            left.len(),
            right.len()
        ));
    }
    let at = left.iter().zip(right).position(|(a, b)| a != b)? / 4;
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes[4 * at..4 * at + 4].try_into().unwrap());
    let (a, b) = (word(left), word(right));
    Some(format!(
        "{name}: word {at} differs: reference {a:#010x} ({}) and merged {b:#010x} ({})",
        f32::from_bits(a),
        f32::from_bits(b)
    ))
}

/// A normalized row: pattern `seed` gives its own picks with high probability.
fn pattern(seed: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 - 1001.0) / 700.0)
        .collect()
}

fn residual(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0)
        .collect()
}

/// Everything the block leaves on the device, as bytes, in a fixed order.
struct Snapshot {
    parts: Vec<(&'static str, Vec<u8>)>,
}

impl Snapshot {
    fn part(&self, name: &str) -> &[u8] {
        &self.parts.iter().find(|(n, _)| *n == name).unwrap().1
    }
}

fn compare(left: &Snapshot, right: &Snapshot, context: &str) {
    assert_eq!(left.parts.len(), right.parts.len());
    for ((name, a), (_, b)) in left.parts.iter().zip(&right.parts) {
        if let Some(message) = first_difference(name, a, b) {
            panic!("{context}: {message}");
        }
    }
}

/// The buffers of one run of a block, on fresh storage.
struct Owned {
    buffers: Vec<DeviceBuffer>,
}

// Indices into `Owned::buffers`.
const RESID_IN: usize = 0;
const RESID_OUT: usize = 1;
const NORM_ROW: usize = 2;
const LOGITS: usize = 3;
const IDS: usize = 4;
const WEIGHTS: usize = 5;
const GATE: usize = 6;
const STATUS: usize = 7;
const GU: usize = 8;
const ACT: usize = 9;
const DOWN: usize = 10;
const SHARED_GU: usize = 11;
const SHARED_ACT: usize = 12;
const SHARED_OUT: usize = 13;
const XH: usize = 14;
const Z: usize = 15;
const FLAG: usize = 16;

impl Owned {
    fn new(projector: &CudaProjector, row: &[f32], moe: &MoeDims) -> Self {
        let h = moe.hidden * F32;
        let logits = router_logits_bytes(crate::kernels::cuda::resident::router::RouterShape {
            m: 1,
            hidden: moe.hidden,
            n_expert: moe.n_expert,
            top_k: moe.top_k,
        })
        .unwrap();
        Self {
            buffers: vec![
                upload(projector, &residual(moe.hidden)),
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
                // The reference sizes of the plan; the merged block uses its own.
                zeros(projector, 2 * moe.top_k * moe.hidden * F32),
                zeros(projector, 2 * moe.top_k * moe.inter.max(moe.hidden) * F32),
                zeros(projector, 4),
            ],
        }
    }

    fn range(&self, index: usize) -> crate::kernels::cuda::DeviceRange<'_> {
        let buffer = &self.buffers[index];
        buffer.range(0, buffer.bytes()).unwrap()
    }

    fn bufs(&self) -> Exl3MoeBufs<'_> {
        let r = |i| self.range(i);
        Exl3MoeBufs {
            route: RouteBufs {
                resid_in: r(RESID_IN),
                resid_out: r(RESID_OUT),
                norm_row: r(NORM_ROW),
                logits: r(LOGITS),
                ids: r(IDS),
                weights: r(WEIGHTS),
                shared_gate: r(GATE),
                status: r(STATUS),
            },
            expert_gu: r(GU),
            expert_act: r(ACT),
            expert_down: r(DOWN),
            shared_gu: r(SHARED_GU),
            shared_act: r(SHARED_ACT),
            shared_out: r(SHARED_OUT),
            scratch: Scratch {
                xh: r(XH),
                z: r(Z),
                flag: r(FLAG),
            },
        }
    }
}

struct Rig {
    projector: CudaProjector,
    checkpoint: Checkpoint,
    table: LayerTable,
    moe: MoeDims,
}

fn rig() -> Rig {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    assert!(device_present("shared in routed"), "a GPU is required");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    let d = dims();
    let table = LayerTable::new(&d, &inventory()).unwrap();
    let moe = MoeDims::new(&d, STEP_ROWS);
    let projector = CudaProjector::new().expect("projector");
    // The kernel the Qwen3.6 start registers; the merged block refuses any other.
    projector
        .context()
        .set_single_row_default(SingleRow::Wide(1));
    Rig {
        projector,
        checkpoint,
        table,
        moe,
    }
}

impl Rig {
    fn weights(&self, layer: usize) -> CudaWeights {
        let reader = PackReader::new(&self.projector, &self.checkpoint, Reserve::default());
        let loaded = moe_weights(&reader, &self.table, layer, &self.moe).expect("moe tensors");
        CudaWeights::from_exl3(loaded).unwrap()
    }

    fn reference(
        &self,
        block: &Exl3MoeWeights<'_>,
        row: &[f32],
        device_ids: bool,
    ) -> Result<Snapshot, EngError> {
        let moe = &self.moe;
        let owned = Owned::new(&self.projector, row, moe);
        let ctx = self.projector.context();
        let stream = ctx.stream()?;
        run_exl3_moe(
            &Launch {
                ctx,
                stream: &stream,
            },
            block,
            &owned.bufs(),
            moe,
            device_ids,
            None,
            false,
        )?;
        stream.synchronize()?;
        let get = |index: usize| {
            let b = &owned.buffers[index];
            read(&self.projector, b, 0, b.bytes())
        };
        Ok(Snapshot {
            parts: vec![
                ("ids", get(IDS)),
                ("pick weights", get(WEIGHTS)),
                ("shared gate", get(GATE)),
                ("routed gate/up rows", get(GU)),
                ("shared gate/up row", get(SHARED_GU)),
                ("routed swiglu rows", get(ACT)),
                ("shared swiglu row", get(SHARED_ACT)),
                ("routed down rows", get(DOWN)),
                ("shared expert output", get(SHARED_OUT)),
                ("combine output", get(RESID_OUT)),
                ("flag", get(FLAG)),
                ("status", get(STATUS)),
            ],
        })
    }

    fn merged(
        &self,
        block: &Exl3MoeWeights<'_>,
        row: &[f32],
        device_ids: bool,
    ) -> Result<Snapshot, EngError> {
        let moe = &self.moe;
        let ctx = self.projector.context();
        let merge = SharedMerge::new(ctx, moe, std::slice::from_ref(block))?;
        let owned = Owned::new(&self.projector, row, moe);
        let stream = ctx.stream()?;
        run_exl3_moe_merged(
            &Launch {
                ctx,
                stream: &stream,
            },
            (block, &owned.bufs(), &merge),
            moe,
            device_ids,
            None,
            false,
        )?;
        stream.synchronize()?;
        let get = |index: usize| {
            let b = &owned.buffers[index];
            read(&self.projector, b, 0, b.bytes())
        };
        let (gu, act, down, zero) = merge.layout_for_tests();
        let (h, i, k) = (moe.hidden * F32, moe.inter * F32, moe.top_k);
        let from =
            |offset: usize, bytes: usize| read(&self.projector, merge.buffer(), offset, bytes);
        let zero_word = from(zero, 4);
        assert_eq!(
            zero_word, [0; 4],
            "the zero word of the shared pairs stays zero"
        );
        Ok(Snapshot {
            parts: vec![
                ("ids", get(IDS)),
                ("pick weights", get(WEIGHTS)),
                ("shared gate", get(GATE)),
                ("routed gate/up rows", from(gu, k * 2 * i)),
                ("shared gate/up row", from(gu + k * 2 * i, 2 * i)),
                ("routed swiglu rows", from(act, k * i)),
                ("shared swiglu row", from(act + k * i, i)),
                ("routed down rows", from(down, k * h)),
                ("shared expert output", from(down + k * h, h)),
                ("combine output", get(RESID_OUT)),
                ("flag", get(FLAG)),
                ("status", get(STATUS)),
            ],
        })
    }
}

fn picks(snapshot: &Snapshot) -> Vec<i32> {
    snapshot
        .part("ids")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "the 3090, held exclusively: loads two layers of the real EXL3 pack; run by name"]
fn shared_in_routed_block_bytes_equal_the_reference_block() {
    let rig = rig();
    let mut distinct = BTreeSet::new();
    for layer in LAYERS {
        let weights = rig.weights(layer);
        let block = Exl3MoeWeights::resolve(&weights, &rig.table, layer, &rig.moe).unwrap();
        for seed in 0..6 {
            let row = pattern(seed, rig.moe.hidden);
            for device_ids in [false, true] {
                let want = rig.reference(&block, &row, device_ids).expect("reference");
                let got = rig.merged(&block, &row, device_ids).expect("merged");
                compare(
                    &want,
                    &got,
                    &format!("layer {layer} pattern {seed} device_ids {device_ids}"),
                );
                assert_eq!(
                    want.part("flag"),
                    [0; 4],
                    "a clean row leaves the flag clear"
                );
                distinct.insert((layer, picks(&want)));
            }
        }
    }
    // At least two different routing patterns were compared on every layer.
    for layer in LAYERS {
        let patterns = distinct.iter().filter(|(l, _)| *l == layer).count();
        assert!(
            patterns >= 2,
            "layer {layer}: {patterns} routing pattern(s)"
        );
    }
    println!(
        "compared {} (layer, picks) patterns byte for byte",
        distinct.len()
    );
}

/// Rows with values the arithmetic flag exists for.
fn special_rows(hidden: usize) -> Vec<(&'static str, Vec<f32>)> {
    let base = pattern(1, hidden);
    let map = |f: &dyn Fn(usize, f32) -> f32| -> Vec<f32> {
        base.iter().enumerate().map(|(i, &v)| f(i, v)).collect()
    };
    vec![
        ("zeros", vec![0.0; hidden]),
        ("negative zeros", vec![-0.0; hidden]),
        (
            "denormals",
            map(&|i, _| if i % 2 == 0 { 1e-40 } else { -1e-40 }),
        ),
        ("tiny", map(&|_, v| v * 1e-30)),
        ("huge", map(&|_, v| v * 1e30)),
        (
            "one infinity",
            map(&|i, v| if i == 5 { f32::INFINITY } else { v }),
        ),
        ("one nan", map(&|i, v| if i == 9 { f32::NAN } else { v })),
        ("max finite", vec![f32::MAX; hidden]),
    ]
}

#[test]
#[ignore = "the 3090, held exclusively: loads two layers of the real EXL3 pack; run by name"]
fn shared_in_routed_special_values_equal_the_reference() {
    let rig = rig();
    for layer in LAYERS {
        let weights = rig.weights(layer);
        let block = Exl3MoeWeights::resolve(&weights, &rig.table, layer, &rig.moe).unwrap();
        for (name, row) in special_rows(rig.moe.hidden) {
            // Device picks only: a non-finite router row makes the host pick read refuse.
            let want = rig.reference(&block, &row, true).expect("reference");
            let got = rig.merged(&block, &row, true).expect("merged");
            compare(&want, &got, &format!("layer {layer} {name}"));
            println!(
                "layer {layer} {name}: flag {:?}, picks {:?}",
                want.part("flag"),
                picks(&want)
            );
        }
    }
}

/// The four scratch areas of one launch of the kernel-level test.
struct Scratch4 {
    xh: DeviceBuffer,
    z: DeviceBuffer,
    out: DeviceBuffer,
    flag: DeviceBuffer,
}

#[derive(Clone, Copy)]
enum Part {
    Xh,
    Z,
    Out,
}

impl Scratch4 {
    fn new(projector: &CudaProjector, rows: usize, k: usize, n: usize) -> Self {
        Self {
            xh: zeros(projector, rows * k * F32),
            z: zeros(projector, rows * n * F32),
            out: zeros(projector, rows * n * F32),
            flag: zeros(projector, 4),
        }
    }

    /// The io of pair `index`, one row of `x`.
    fn io<'a>(
        &'a self,
        x: crate::kernels::cuda::DeviceRange<'a>,
        k: usize,
        n: usize,
        index: usize,
    ) -> Exl3Io<'a> {
        Exl3Io {
            x,
            xh: self.xh.range(index * k * F32, k * F32).unwrap(),
            z: self.z.range(index * n * F32, n * F32).unwrap(),
            out: self.out.range(index * n * F32, n * F32).unwrap(),
            flag: self.flag.range(0, 4).unwrap(),
            rows: 1,
            fused: false,
        }
    }

    fn rows(
        &self,
        projector: &CudaProjector,
        part: Part,
        first: usize,
        count: usize,
        width: usize,
    ) -> Vec<u8> {
        let buffer = match part {
            Part::Xh => &self.xh,
            Part::Z => &self.z,
            Part::Out => &self.out,
        };
        read(projector, buffer, first * width * F32, count * width * F32)
    }
}

#[test]
#[ignore = "the 3090, held exclusively: loads one layer of the real EXL3 pack; run by name"]
fn shared_in_routed_pair_equals_the_grouped_pair_on_real_tensors() {
    let rig = rig();
    let layer = LAYERS[0];
    let weights = rig.weights(layer);
    let block = Exl3MoeWeights::resolve(&weights, &rig.table, layer, &rig.moe).unwrap();
    let moe = &rig.moe;
    let ctx = rig.projector.context();
    let mut rows: Vec<(String, Vec<f32>)> = (0..4)
        .map(|seed| (format!("pattern {seed}"), pattern(seed, moe.hidden)))
        .collect();
    rows.extend(
        special_rows(moe.hidden)
            .into_iter()
            .filter(|(name, _)| !name.contains("nan"))
            .map(|(name, row)| (name.to_owned(), row)),
    );
    for (name, row) in rows {
        let x = upload(&rig.projector, &row);
        let xr = x.range(0, x.bytes()).unwrap();
        let pairs = 2 * moe.top_k + 2;
        let ids_bytes: Vec<u8> = (0..moe.top_k as i32)
            .map(|s| (s * 31 + 5) % moe.n_expert as i32)
            .flat_map(i32::to_le_bytes)
            .collect();
        let ids = ctx.upload(&ids_bytes).unwrap();
        let zero = zeros(&rig.projector, 4);
        let (h, i) = (moe.hidden, moe.inter);
        // Each case gets its own scratch so nothing is shared between the launches.
        let grouped = Scratch4::new(&rig.projector, 2, h, i);
        let wide = Scratch4::new(&rig.projector, pairs, h, i);
        let narrow = Scratch4::new(&rig.projector, 2 * moe.top_k, h, i);
        let stream = ctx.stream().unwrap();
        // A: shared gate and up through the grouped launch.
        let shared: [&Exl3Weight; 2] = [block.shared_gate, block.shared_up];
        let views: Vec<_> = shared.iter().map(|w| w.view().unwrap()).collect();
        let calls: Vec<_> = views
            .iter()
            .enumerate()
            .map(|(index, view)| view.linear_call(&grouped.io(xr, h, i, index)))
            .collect();
        ctx.exl3_grouped_step(&stream, &calls).unwrap();
        // B: the 18-pair indexed launch.
        let spans = [block.gate.span().unwrap(), block.up.span().unwrap()];
        let mut calls = Vec::new();
        for slot in 0..moe.top_k {
            for span in &spans {
                let index = calls.len();
                calls.push(span.indexed_call(&wide.io(xr, h, i, index), full(&ids), slot as u32));
            }
        }
        for view in &views {
            let index = calls.len();
            calls.push(view.indexed_call(&wide.io(xr, h, i, index), full(&zero), 0));
        }
        assert_eq!(calls.len(), 18);
        ctx.exl3_grouped_indexed_wide(&stream, &calls, ctx.exl3_indexed_cols(i).unwrap())
            .unwrap();
        // C: the routed pairs alone in the default 16-pair launch.
        let mut calls = Vec::new();
        for slot in 0..moe.top_k {
            for span in &spans {
                let index = calls.len();
                calls.push(span.indexed_call(&narrow.io(xr, h, i, index), full(&ids), slot as u32));
            }
        }
        ctx.exl3_grouped_indexed(&stream, &calls, ctx.exl3_indexed_cols(i).unwrap())
            .unwrap();
        stream.synchronize().unwrap();
        let routed = 2 * moe.top_k;
        let checks = [
            (
                "shared rotated input",
                grouped.rows(&rig.projector, Part::Xh, 0, 2, h),
                wide.rows(&rig.projector, Part::Xh, routed, 2, h),
            ),
            (
                "shared partial",
                grouped.rows(&rig.projector, Part::Z, 0, 2, i),
                wide.rows(&rig.projector, Part::Z, routed, 2, i),
            ),
            (
                "shared output",
                grouped.rows(&rig.projector, Part::Out, 0, 2, i),
                wide.rows(&rig.projector, Part::Out, routed, 2, i),
            ),
            (
                "routed rotated inputs",
                narrow.rows(&rig.projector, Part::Xh, 0, routed, h),
                wide.rows(&rig.projector, Part::Xh, 0, routed, h),
            ),
            (
                "routed partials",
                narrow.rows(&rig.projector, Part::Z, 0, routed, i),
                wide.rows(&rig.projector, Part::Z, 0, routed, i),
            ),
            (
                "routed outputs",
                narrow.rows(&rig.projector, Part::Out, 0, routed, i),
                wide.rows(&rig.projector, Part::Out, 0, routed, i),
            ),
        ];
        for (what, reference, merged) in checks {
            if let Some(message) = first_difference(what, &reference, &merged) {
                panic!("{name}: {message}");
            }
        }
        // The flag of the merged launch is the union of the two reference launches' flags.
        let flag = |s: &Scratch4| read(&rig.projector, &s.flag, 0, 4);
        let union: Vec<u8> = flag(&grouped)
            .iter()
            .zip(flag(&narrow))
            .map(|(a, c)| a | c)
            .collect();
        assert_eq!(flag(&wide), union, "{name}: flag");
    }
}

#[test]
#[ignore = "the 3090, held exclusively: loads one layer of the real EXL3 pack; run by name"]
fn shared_in_routed_graph_has_seven_fewer_kernel_nodes() {
    let rig = rig();
    let layer = LAYERS[0];
    let weights = rig.weights(layer);
    let block = Exl3MoeWeights::resolve(&weights, &rig.table, layer, &rig.moe).unwrap();
    let ctx = rig.projector.context();
    let owned = Owned::new(&rig.projector, &pattern(0, rig.moe.hidden), &rig.moe);
    let merge = SharedMerge::new(ctx, &rig.moe, std::slice::from_ref(&block)).unwrap();
    let stream = ctx.stream().unwrap();
    let capture = |merged: bool| {
        let plan = CapturePlan {
            pins: &[],
            holds: Vec::new(),
        };
        let bufs = owned.bufs();
        let launch = Launch {
            ctx,
            stream: &stream,
        };
        let (graph, ()) = ctx
            .capture(&stream, plan, || {
                if merged {
                    run_exl3_moe_merged(
                        &launch,
                        (&block, &bufs, &merge),
                        &rig.moe,
                        true,
                        None,
                        false,
                    )
                } else {
                    run_exl3_moe(&launch, &block, &bufs, &rig.moe, true, None, false)
                }
            })
            .expect("capture");
        graph.nodes()
    };
    let reference = capture(false);
    let merged = capture(true);
    println!("kernel nodes per MoE layer: reference {reference}, merged {merged}");
    assert_eq!(
        reference - merged,
        7,
        "gate/up, swiglu and down of the shared expert fold in"
    );
    assert!(merged < reference);
}
