//! The split MoE block on two real layers of the full pack, against the all-resident block
//! (`qwen36::cuda::exl3_moe::run_exl3_moe` over a bank of every expert) and against an fp64
//! evaluation of the stored matrices.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names the local pack directory (the full 3.05 bpw pack, or a copy
//! with the headers of every shard and the payload of the layers' MoE tensors);
//! `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` picks the first layer (3 by default), the second is the next
//! one. The bank has two rows (bank layer 0 is the first pack layer, bank layer 1 the second) with
//! 16 slots each, so the block runs on a layer other than 0, with a trunk other than the first.
//! The routed experts of the all-resident runs are uploaded as separate banks.
//!
//! Cases, each on the real router's picks for a fixed input row of each layer:
//! - a bank that holds six of the row's ten picks: a mixed pass with four misses, then the same row
//!   again with no miss;
//! - a bank that holds none of the row's picks: ten misses, the resident pass is all filler, then
//!   the same row again.
//!
//! Each case runs with the finite-filler check on and again, on a fresh bank, with it off. With it
//! off the host does not read the stream between the resident pass and the bank's copies, so the
//! copies overlap the resident pass: the default path.
//!
//! Checked: a layer without a miss writes the bytes of the all-resident block; a layer with misses
//! is within a relative L2 of 1e-6 of it (the sums are associated in two groups, every product is
//! the same); both are within the block's bound of fp64; the bank holds every pick afterwards.

use std::path::PathBuf;

use super::{
    bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::DeviceBank,
    },
    bank_source::PackRecordSource,
    moe_block::{SplitMoe, SplitMoeTrunk},
};
use crate::{
    EngError,
    family::{
        flash_next::{
            load::{exl3_pack::inspect_flash_pack, model_shape, pack_files::FlashPackFiles},
            moe::exl3_cuda_tests::{
                OUTPUT_BOUND, Reference, Router, floats, input, keys, reference, relative_l2,
                upload,
            },
        },
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, CudaWeights},
        qwen36::cuda::{
            exl3_moe::{Exl3MoeBufs, Exl3MoeWeights, Scratch, run_exl3_moe},
            moe::{Launch, MoeDims, RouteBufs},
        },
    },
    kernels::cuda::{
        DeviceBuffer, device_present,
        resident::router::{RouterShape, router_logits_bytes},
    },
    weights::exl3::Checkpoint,
};

const F32: usize = 4;
const SLOTS: usize = 16;
/// The two groupings of the sums differ only by association: a handful of float roundings.
const SPLIT_BOUND: f64 = 1e-6;
/// The ring branch computes every projection through the job-table tile kernels, which stage
/// the F32 activation as F16 operands (documented operand error 2^-11, `exl3.cu`'s
/// midrows_projection; the hi/lo opt-in cuts it to ~2^-22), while the all-resident and fp64
/// references run the F32-operand flat launches. Measured 1.5e-4 to 2.1e-4 relative L2,
/// identical whether every pick sits in the bank, in the ring, or split over both (the
/// sources and their order change nothing; FIFI_FLASH_NEXT_RING_HELD arms those variants),
/// and identical with the fused pair launch on and off. The bound keeps ~2x headroom over
/// the measured spread; the resident comparison cannot tighten to SPLIT_BOUND because the
/// difference is the launch family, not the sources.
const RING_BOUND: f64 = 4e-4;

fn pack_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"))
}

fn moe_layer() -> usize {
    std::env::var("FIFI_FLASH_NEXT_EXL3_MOE_LAYER")
        .map_or(3, |value| value.parse().expect("a layer number"))
}

/// Bank layer `i` holds pack layer `layers[i]`.
struct Layers<'a, 'b> {
    inner: &'b PackRecordSource<'a>,
    layers: Vec<usize>,
}

impl RecordSource for Layers<'_, '_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        self.inner.record(self.layers[layer], expert, out)
    }
}

/// The buffers of one block call, zeroed, with `row` as the normalized input.
struct Rig {
    owned: Vec<DeviceBuffer>,
}

impl Rig {
    fn new(projector: &CudaProjector, moe: &MoeDims, row: &[f32]) -> Result<Self, EngError> {
        let ctx = projector.context();
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        let (h, k) = (moe.hidden * F32, moe.top_k);
        let logits = router_logits_bytes(RouterShape {
            m: 1,
            hidden: moe.hidden,
            n_expert: moe.n_expert,
            top_k: k,
        })?;
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let owned = vec![
            zeros(h)?,
            zeros(h)?,
            ctx.upload(&bytes)?,
            zeros(logits)?,
            zeros(k * 4)?,
            zeros(k * F32)?,
            zeros(F32)?,
            zeros(4)?,
            zeros(k * 2 * moe.inter * F32)?,
            zeros(k * moe.inter * F32)?,
            zeros(k * h)?,
            zeros(2 * moe.shared_inter * F32)?,
            zeros(moe.shared_inter * F32)?,
            zeros(h)?,
            zeros(2 * k * moe.hidden * F32)?,
            zeros(2 * k * moe.inter.max(moe.hidden) * F32)?,
            zeros(4)?,
        ];
        Ok(Self { owned })
    }

    fn bufs(&self) -> Result<Exl3MoeBufs<'_>, EngError> {
        let r = |i: usize| self.owned[i].range(0, self.owned[i].bytes());
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: r(0)?,
                resid_out: r(1)?,
                norm_row: r(2)?,
                logits: r(3)?,
                ids: r(4)?,
                weights: r(5)?,
                shared_gate: r(6)?,
                status: r(7)?,
            },
            expert_gu: r(8)?,
            expert_act: r(9)?,
            expert_down: r(10)?,
            shared_gu: r(11)?,
            shared_act: r(12)?,
            shared_out: r(13)?,
            scratch: Scratch {
                xh: r(14)?,
                z: r(15)?,
                flag: r(16)?,
            },
        })
    }

    fn out_bytes(&self, projector: &CudaProjector) -> Vec<u8> {
        let stream = projector.context().stream().expect("stream");
        let mut bytes = vec![0u8; self.owned[1].bytes()];
        self.owned[1]
            .read_at(&stream, 0, &mut bytes)
            .expect("read the output");
        bytes
    }
}

fn raw<'w>(weights: &'w CudaWeights, name: &str) -> &'w DeviceBuffer {
    match weights.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => buffer,
        _ => panic!("{name} is not a raw tensor"),
    }
}

fn exl3<'w>(
    weights: &'w CudaWeights,
    name: &str,
) -> &'w crate::family::project::cuda::exl3::Exl3Weight {
    match weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => weight,
        _ => panic!("{name} is not an EXL3 matrix"),
    }
}

/// A bank with one row per entry of `held` (slot `s` of row `r` holds `held[r][s]`) over the pack
/// layers `layers`.
fn build_bank(
    projector: &CudaProjector,
    source: &PackRecordSource<'_>,
    layers: &[usize],
    held: &[Vec<u16>],
) -> DeviceBank {
    let record = source.layout().record_bytes();
    let slots = held[0].len();
    let geometry = BankGeometry::new(layers.len(), 512, 10, slots, record).expect("geometry");
    let rows: Vec<Vec<i32>> = held
        .iter()
        .map(|held| {
            let mut row = vec![-1i32; 512];
            for (slot, &expert) in held.iter().enumerate() {
                row[usize::from(expert)] = slot as i32;
            }
            row
        })
        .collect();
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let counts: Vec<Vec<u64>> = layers.iter().map(|_| (0..512u64).rev().collect()).collect();
    let policy = ReplaceOnMissLru::from_fill(geometry, &row_refs, &counts, None).expect("the fill");
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).expect("state");
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    DeviceBank::build(
        projector.context(),
        state,
        &Layers {
            inner: source,
            layers: layers.to_vec(),
        },
        &plan,
    )
    .expect("build the bank")
}

/// `count` experts that are not in `avoid`, lowest ids first.
fn others(avoid: &[usize], count: usize) -> Vec<u16> {
    (0..512u16)
        .filter(|&e| !avoid.contains(&usize::from(e)))
        .take(count)
        .collect()
}

/// One pack layer's uploads and its references on its own input row.
struct Prepared {
    row: Vec<f32>,
    weights: CudaWeights,
    keys: Vec<(Role, String)>,
    want: Reference,
    resident: Vec<u8>,
}

impl Prepared {
    fn key(&self, role: Role) -> &str {
        self.keys
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .unwrap_or_else(|| panic!("no key for {role:?}"))
    }

    fn trunk(&self) -> SplitMoeTrunk<'_> {
        SplitMoeTrunk {
            router: raw(&self.weights, self.key(Role::FfnGateInp)),
            router_shared: raw(&self.weights, self.key(Role::FfnGateInpShexp)),
            shared_gate: exl3(&self.weights, self.key(Role::FfnGateShexp)),
            shared_up: exl3(&self.weights, self.key(Role::FfnUpShexp)),
            shared_down: exl3(&self.weights, self.key(Role::FfnDownShexp)),
        }
    }

    fn picks(&self) -> &[usize] {
        &self.want.ids
    }
}

fn prepare(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    (layer, seed): (usize, usize),
    moe: &MoeDims,
    launch: &Launch<'_>,
) -> Prepared {
    let router = Router::read(checkpoint, layer, moe).expect("router");
    let weights = upload(projector, checkpoint, (layer, &router), moe).expect("upload");
    let keys = keys(layer);
    let row = input(moe.hidden, seed);
    let want = reference(checkpoint, (layer, &router), moe, &row).expect("fp64 reference");
    // The all-resident output on the same row, one launch over every expert.
    let resident = {
        let block = Exl3MoeWeights::resolve_by(&weights, layer, moe, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
        })
        .expect("the all-resident block resolves");
        let rig = Rig::new(projector, moe, &row).expect("rig");
        run_exl3_moe(
            launch,
            &block,
            &rig.bufs().expect("bufs"),
            moe,
            true,
            None,
            false,
        )
        .expect("block");
        launch.stream.synchronize().expect("sync");
        rig.out_bytes(projector)
    };
    let error = relative_l2(&floats(&resident), &want.out);
    assert!(error < OUTPUT_BOUND, "layer {layer} all-resident {error}");
    Prepared {
        row,
        weights,
        keys,
        want,
        resident,
    }
}

#[test]
#[ignore = "needs a GPU (about 4 GiB), about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_real_layer_split_over_a_partial_bank_matches_the_all_resident_block_and_fp64() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let moe = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    assert_eq!((moe.n_expert, moe.top_k), (512, 10));
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let prepared: Vec<Prepared> = layers
        .iter()
        .enumerate()
        .map(|(index, &layer)| prepare(&projector, checkpoint, (layer, index), &moe, &launch))
        .collect();
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    for verify in [true, false] {
        for case in ["mixed", "forced miss"] {
            // Per bank layer: the experts its slots hold and the misses its row will cause.
            let held: Vec<Vec<u16>> = prepared
                .iter()
                .map(|p| match case {
                    // Six of the ten picks held, ten other experts: four misses.
                    "mixed" => {
                        let mut held: Vec<u16> =
                            p.picks().iter().take(6).map(|&e| e as u16).collect();
                        held.extend(others(p.picks(), SLOTS - 6));
                        held
                    }
                    // None held: ten misses, the resident pass is all filler.
                    _ => others(p.picks(), SLOTS),
                })
                .collect();
            let expected_misses: Vec<usize> = prepared
                .iter()
                .zip(&held)
                .map(|(p, held)| {
                    p.picks()
                        .iter()
                        .filter(|&&pick| !held.contains(&(pick as u16)))
                        .count()
                })
                .collect();
            let bank = build_bank(&projector, &source, &layers, &held);
            let trunks = prepared.iter().map(Prepared::trunk).collect();
            let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &moe).expect("split");
            split.verify_finite_filler(verify);
            let mut errors = Vec::new();
            for run in 0..2 {
                // Bank layer 1 first, then layer 0.
                for bank_layer in [1usize, 0] {
                    let p = &prepared[bank_layer];
                    let expect = if run == 0 {
                        expected_misses[bank_layer]
                    } else {
                        0
                    };
                    let rig = Rig::new(&projector, &moe, &p.row).expect("rig");
                    let before = split.bank().state().stats().misses;
                    split
                        .run(&launch, bank_layer, &rig.bufs().expect("bufs"), &moe)
                        .expect("the split block");
                    stream.synchronize().expect("sync");
                    let got = rig.out_bytes(&projector);
                    let misses = (split.bank().state().stats().misses - before) as usize;
                    let tag = format!(
                        "{case}, filler check {}, run {run}, bank layer {bank_layer} (pack layer {})",
                        if verify { "on" } else { "off" },
                        layers[bank_layer]
                    );
                    assert_eq!(misses, expect, "{tag}: misses");
                    let resident_error = relative_l2(&floats(&got), &floats_f64(&p.resident));
                    let fp64_error = relative_l2(&floats(&got), &p.want.out);
                    println!(
                        "{tag}: {misses} misses, vs all-resident {resident_error:.3e}, vs fp64 \
                         {fp64_error:.3e}"
                    );
                    assert!(fp64_error < OUTPUT_BOUND, "{tag}: fp64 {fp64_error}");
                    if expect == 0 {
                        assert_eq!(got, p.resident, "{tag}: no miss writes the resident bytes");
                    } else {
                        assert!(resident_error < SPLIT_BOUND, "{tag}: {resident_error}");
                    }
                    errors.push(resident_error);
                }
            }
            for (bank_layer, p) in prepared.iter().enumerate() {
                for &pick in p.picks() {
                    assert!(
                        split.bank().state().slot_row(bank_layer)[pick] >= 0,
                        "{case}: pick {pick} of bank layer {bank_layer} is in the bank"
                    );
                }
            }
        }
    }
}

fn floats_f64(bytes: &[u8]) -> Vec<f64> {
    floats(bytes).into_iter().map(f64::from).collect()
}

/// The buffers of one `run_rows` call, zeroed, with `rows` as the normalized inputs. The
/// grouped launches chunk at the same capacity as one row's, restarting at scratch row 0, so
/// the EXL3 scratch keeps its one-row sizing.
struct RigRows {
    owned: Vec<DeviceBuffer>,
    hidden: usize,
}

impl RigRows {
    fn new(projector: &CudaProjector, moe: &MoeDims, rows: &[Vec<f32>]) -> Result<Self, EngError> {
        let m = rows.len();
        let ctx = projector.context();
        let zeros = |bytes: usize| ctx.upload(&vec![0u8; bytes]);
        let (h, k) = (moe.hidden * F32, moe.top_k);
        let one = router_logits_bytes(RouterShape {
            m: 1,
            hidden: moe.hidden,
            n_expert: moe.n_expert,
            top_k: k,
        })?;
        let mut norm = Vec::with_capacity(m * h);
        for row in rows {
            norm.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        let owned = vec![
            zeros(m * h)?,
            zeros(m * h)?,
            ctx.upload(&norm)?,
            zeros(m * one)?,
            zeros(m * k * 4)?,
            zeros(m * k * F32)?,
            zeros(m * F32)?,
            zeros(4)?,
            zeros(m * k * 2 * moe.inter * F32)?,
            zeros(m * k * moe.inter * F32)?,
            zeros(m * k * h)?,
            zeros(m * 2 * moe.shared_inter * F32)?,
            zeros(m * moe.shared_inter * F32)?,
            zeros(m * h)?,
            // A source's gate/up launch stages one xh row a pair, and the whole chunk can be
            // one source's: m·k rows, like the down windows through z.
            zeros(m * k * moe.hidden * F32)?,
            zeros(m * k * moe.inter.max(moe.hidden) * F32)?,
            zeros(4)?,
        ];
        Ok(Self {
            owned,
            hidden: moe.hidden * F32,
        })
    }

    fn bufs(&self) -> Result<Exl3MoeBufs<'_>, EngError> {
        let r = |i: usize| self.owned[i].range(0, self.owned[i].bytes());
        Ok(Exl3MoeBufs {
            route: RouteBufs {
                resid_in: r(0)?,
                resid_out: r(1)?,
                norm_row: r(2)?,
                logits: r(3)?,
                ids: r(4)?,
                weights: r(5)?,
                shared_gate: r(6)?,
                status: r(7)?,
            },
            expert_gu: r(8)?,
            expert_act: r(9)?,
            expert_down: r(10)?,
            shared_gu: r(11)?,
            shared_act: r(12)?,
            shared_out: r(13)?,
            scratch: Scratch {
                xh: r(14)?,
                z: r(15)?,
                flag: r(16)?,
            },
        })
    }

    fn out_row(&self, projector: &CudaProjector, r: usize) -> Vec<u8> {
        let stream = projector.context().stream().expect("stream");
        let mut bytes = vec![0u8; self.owned[1].bytes()];
        self.owned[1]
            .read_at(&stream, 0, &mut bytes)
            .expect("read the outputs");
        bytes[r * self.hidden..][..self.hidden].to_vec()
    }
}

/// `run_rows` with the prefill ring funded: every pick reads its bank slot or its ring slot,
/// the bank's miss counter stays flat (the ring path never steps or admits the bank), and
/// every row matches the all-resident block and fp64 within RING_BOUND run after run (the
/// ring re-arms per chunk like the driver cadence). RING_BOUND, not SPLIT_BOUND, because the
/// ring branch's job-table tile kernels carry the F16 operand staging the flat-launch
/// references do not — see the bound's doc for the measurements that isolated it.
#[test]
#[ignore = "needs a GPU (about 3 GiB of ring and bank, about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_chunk_over_the_prefill_ring_matches_the_all_resident_block() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    // 49 rows: three gate/up pair batches (24 + 24 + 1) and a multi-group down window —
    // wider than any single launch bound of the shared expert.
    // 49 rows: three gate/up pair batches (24 + 24 + 1) and a multi-group down window —
    // wider than any single launch bound of the shared expert.
    let m = 49;
    let chunk = MoeDims { rows: m, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    // Per bank layer: uploads, the chunk's rows, and per-row fp64 and all-resident references.
    struct Arm {
        row_inputs: Vec<Vec<f32>>,
        weights: CudaWeights,
        keys: Vec<(Role, String)>,
        wants: Vec<Reference>,
        residents: Vec<Vec<u8>>,
    }
    let mut arms: Vec<Arm> = Vec::with_capacity(layers.len());
    for (index, &layer) in layers.iter().enumerate() {
        let router = Router::read(checkpoint, layer, &one_row).expect("router");
        let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
        let keys = keys(layer);
        let row_inputs: Vec<Vec<f32>> = (0..m)
            .map(|seed| input(one_row.hidden, seed + 7 * (index + 1)))
            .collect();
        let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
        })
        .expect("the all-resident block resolves");
        let mut wants = Vec::with_capacity(m);
        let mut residents = Vec::with_capacity(m);
        for row in &row_inputs {
            wants.push(
                reference(checkpoint, (layer, &router), &one_row, row).expect("fp64 reference"),
            );
            let rig = Rig::new(&projector, &one_row, row).expect("rig");
            run_exl3_moe(
                &launch,
                &block,
                &rig.bufs().expect("bufs"),
                &one_row,
                true,
                None,
                false,
            )
            .expect("the all-resident block");
            launch.stream.synchronize().expect("sync");
            residents.push(rig.out_bytes(&projector));
        }
        arms.push(Arm {
            row_inputs,
            weights,
            keys,
            wants,
            residents,
        });
    }

    // The bank holds the first `held_of` of row 0's picks per layer, padded with other experts:
    // the default 6 leaves row 0 mixed; the ring-tolerance measurements hold every pick
    // (FIFI_FLASH_NEXT_RING_HELD=10, the all-bank arm) or none (0, the all-ring arm).
    let held_of: usize = std::env::var("FIFI_FLASH_NEXT_RING_HELD")
        .map_or(6, |v| v.parse().expect("a held-pick count"));
    let held: Vec<Vec<u16>> = arms
        .iter()
        .map(|arm| {
            let mut held: Vec<u16> = arm.wants[0]
                .ids
                .iter()
                .take(held_of)
                .map(|&e| e as u16)
                .collect();
            held.extend(others(&arm.wants[0].ids, SLOTS - held.len()));
            held
        })
        .collect();
    let bank = build_bank(&projector, &source, &layers, &held);
    let trunks: Vec<SplitMoeTrunk<'_>> = arms
        .iter()
        .map(|arm| {
            let key = |role: Role| {
                arm.keys
                    .iter()
                    .find(|(candidate, _)| *candidate == role)
                    .map(|(_, key)| key.as_str())
                    .unwrap_or_else(|| panic!("no key for {role:?}"))
            };
            SplitMoeTrunk {
                router: raw(&arm.weights, key(Role::FfnGateInp)),
                router_shared: raw(&arm.weights, key(Role::FfnGateInpShexp)),
                shared_gate: exl3(&arm.weights, key(Role::FfnGateShexp)),
                shared_up: exl3(&arm.weights, key(Role::FfnUpShexp)),
                shared_down: exl3(&arm.weights, key(Role::FfnDownShexp)),
            }
        })
        .collect();
    let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &chunk).expect("split");
    let streamed = shape.experts() - SLOTS;
    let slots = split.plan_ring(ctx, streamed).expect("fund the ring");
    assert!(slots >= streamed, "the ring holds one full streamed layer");
    split.begin_prefill();

    // One chunk per run, every layer of it in order — the driver cadence the ring's free
    // cursor assumes.
    for run in 0..2 {
        for (bank_layer, arm) in arms.iter().enumerate() {
            if bank_layer == 0 {
                split.begin_chunk();
            }
            let rig = RigRows::new(&projector, &chunk, &arm.row_inputs).expect("rig");
            let before = split.bank().state().stats().misses;
            split
                .run_rows(&launch, bank_layer, &rig.bufs().expect("bufs"), &chunk)
                .expect("the ring block");
            stream.synchronize().expect("sync");
            let misses = (split.bank().state().stats().misses - before) as usize;
            let tag = format!(
                "bank layer {bank_layer} (pack layer {}), run {run}",
                layers[bank_layer]
            );
            // The ring path never steps or admits the bank: the miss counter cannot move.
            assert_eq!(misses, 0, "{tag}: the ring path does not touch the bank");
            for r in 0..m {
                let got = rig.out_row(&projector, r);
                let resident_error = relative_l2(&floats(&got), &floats_f64(&arm.residents[r]));
                let fp64_error = relative_l2(&floats(&got), &arm.wants[r].out);
                println!(
                    "{tag}, row {r}: vs all-resident {resident_error:.3e}, vs fp64 {fp64_error:.3e}"
                );
                assert!(fp64_error < RING_BOUND, "{tag}, row {r}: fp64 {fp64_error}");
                assert!(
                    resident_error < RING_BOUND,
                    "{tag}, row {r}: all-resident {resident_error}"
                );
            }
        }
    }
}

fn a_chunk_of_rows_over_a_partial_bank_matches_the_all_resident_block() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first, first + 1];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    let m = 8;
    let chunk = MoeDims { rows: m, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    // Per bank layer: uploads, the chunk's rows, and per-row fp64 and all-resident references.
    struct Arm {
        row_inputs: Vec<Vec<f32>>,
        weights: CudaWeights,
        keys: Vec<(Role, String)>,
        wants: Vec<Reference>,
        residents: Vec<Vec<u8>>,
    }
    let mut arms: Vec<Arm> = Vec::with_capacity(layers.len());
    for (index, &layer) in layers.iter().enumerate() {
        let router = Router::read(checkpoint, layer, &one_row).expect("router");
        let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
        let keys = keys(layer);
        let row_inputs: Vec<Vec<f32>> = (0..m)
            .map(|seed| input(one_row.hidden, seed + 7 * (index + 1)))
            .collect();
        let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
        })
        .expect("the all-resident block resolves");
        let mut wants = Vec::with_capacity(m);
        let mut residents = Vec::with_capacity(m);
        for row in &row_inputs {
            wants.push(
                reference(checkpoint, (layer, &router), &one_row, row).expect("fp64 reference"),
            );
            let rig = Rig::new(&projector, &one_row, row).expect("rig");
            run_exl3_moe(
                &launch,
                &block,
                &rig.bufs().expect("bufs"),
                &one_row,
                true,
                None,
                false,
            )
            .expect("the all-resident block");
            launch.stream.synchronize().expect("sync");
            residents.push(rig.out_bytes(&projector));
        }
        arms.push(Arm {
            row_inputs,
            weights,
            keys,
            wants,
            residents,
        });
    }

    // The bank holds six of row 0's picks per layer: row 0 is mixed, every later row misses.
    let held: Vec<Vec<u16>> = arms
        .iter()
        .map(|arm| {
            let mut held: Vec<u16> = arm.wants[0].ids.iter().take(6).map(|&e| e as u16).collect();
            held.extend(others(&arm.wants[0].ids, SLOTS - 6));
            held
        })
        .collect();
    let bank = build_bank(&projector, &source, &layers, &held);
    let trunks: Vec<SplitMoeTrunk<'_>> = arms
        .iter()
        .map(|arm| {
            let key = |role: Role| {
                arm.keys
                    .iter()
                    .find(|(candidate, _)| *candidate == role)
                    .map(|(_, key)| key.as_str())
                    .unwrap_or_else(|| panic!("no key for {role:?}"))
            };
            SplitMoeTrunk {
                router: raw(&arm.weights, key(Role::FfnGateInp)),
                router_shared: raw(&arm.weights, key(Role::FfnGateInpShexp)),
                shared_gate: exl3(&arm.weights, key(Role::FfnGateShexp)),
                shared_up: exl3(&arm.weights, key(Role::FfnUpShexp)),
                shared_down: exl3(&arm.weights, key(Role::FfnDownShexp)),
            }
        })
        .collect();
    let mut split = SplitMoe::new(ctx, trunks, bank, &layout, &chunk).expect("split");

    for (bank_layer, arm) in arms.iter().enumerate() {
        for run in 0..2 {
            let rig = RigRows::new(&projector, &chunk, &arm.row_inputs).expect("rig");
            let before = split.bank().state().stats().misses;
            split
                .run_rows(&launch, bank_layer, &rig.bufs().expect("bufs"), &chunk)
                .expect("the chunked block");
            stream.synchronize().expect("sync");
            let misses = (split.bank().state().stats().misses - before) as usize;
            let tag = format!(
                "bank layer {bank_layer} (pack layer {}), run {run}",
                layers[bank_layer]
            );
            // With about `m * top_k` distinct picks against `SLOTS` slots, BOTH chunks miss:
            // even the second finds most picks evicted by the last rows of the first.
            assert!(
                misses > 0,
                "{tag}: the chunk misses under eviction pressure"
            );
            for r in 0..m {
                let got = rig.out_row(&projector, r);
                let resident_error = relative_l2(&floats(&got), &floats_f64(&arm.residents[r]));
                let fp64_error = relative_l2(&floats(&got), &arm.wants[r].out);
                println!(
                    "{tag}, row {r}: vs all-resident {resident_error:.3e}, vs fp64 {fp64_error:.3e}"
                );
                assert!(
                    fp64_error < OUTPUT_BOUND,
                    "{tag}, row {r}: fp64 {fp64_error}"
                );
                assert!(
                    resident_error < SPLIT_BOUND,
                    "{tag}, row {r}: all-resident {resident_error}"
                );
            }
        }
    }
}

/// `run_rows` when the chunk's distinct picks fit the bank (2 rows against 20 slots): the
/// union admit runs and every row rides the flat resident pass — no per-row `step`, which the
/// pass counter proves (decode passes stay flat). Every row matches the all-resident block
/// within the split bound, every pick of the union is resident afterwards, and the same chunk
/// again changes nothing.
#[test]
#[ignore = "needs a GPU (about 3 GiB), about 3 GiB of pinned host memory and FIFI_FLASH_NEXT_EXL3_PACK"]
fn a_chunk_whose_picks_fit_the_bank_admits_the_union_in_one_pass() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let layer = moe_layer();
    let layers = vec![layer];
    let files = FlashPackFiles::open(&pack_dir()).expect("open the pack");
    let checkpoint = &files.checkpoint;
    inspect_flash_pack(&checkpoint.source, &checkpoint.model_config)
        .expect("the pack matches its config");
    let (model, _) = model_shape(&checkpoint.model_config).expect("the model shape");
    let shape = model.moe;
    let one_row = MoeDims {
        rows: 1,
        hidden: model.hyper.hidden(),
        n_expert: shape.experts(),
        top_k: shape.top_k(),
        inter: shape.inter(),
        shared_inter: shape.shared_inter(),
    };
    assert_eq!((one_row.n_expert, one_row.top_k), (512, 10));
    let m = 2;
    let chunk = MoeDims { rows: m, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    let router = Router::read(checkpoint, layer, &one_row).expect("router");
    let weights = upload(&projector, checkpoint, (layer, &router), &one_row).expect("upload");
    let keys = keys(layer);
    let row_inputs: Vec<Vec<f32>> = (0..m)
        .map(|seed| input(one_row.hidden, seed + 31))
        .collect();
    let block = Exl3MoeWeights::resolve_by(&weights, layer, &one_row, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the all-resident block resolves");
    let mut wants = Vec::with_capacity(m);
    let mut residents = Vec::with_capacity(m);
    for row in &row_inputs {
        wants.push(reference(checkpoint, (layer, &router), &one_row, row).expect("fp64 reference"));
        let rig = Rig::new(&projector, &one_row, row).expect("rig");
        run_exl3_moe(
            &launch,
            &block,
            &rig.bufs().expect("bufs"),
            &one_row,
            true,
            None,
            false,
        )
        .expect("the all-resident block");
        launch.stream.synchronize().expect("sync");
        residents.push(rig.out_bytes(&projector));
    }

    // The fill holds all 10 of row 0's picks of 20 slots; the chunk's union (at most 20
    // distinct) always fits, so run_rows takes the union-admit path, never the per-row tail.
    let mut held: Vec<u16> = wants[0].ids.iter().take(10).map(|&e| e as u16).collect();
    held.extend(others(&wants[0].ids, 10));
    let bank = build_bank(&projector, &source, &layers, &[held]);
    let key = |role: Role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .unwrap_or_else(|| panic!("no key for {role:?}"))
    };
    let trunk = SplitMoeTrunk {
        router: raw(&weights, key(Role::FfnGateInp)),
        router_shared: raw(&weights, key(Role::FfnGateInpShexp)),
        shared_gate: exl3(&weights, key(Role::FfnGateShexp)),
        shared_up: exl3(&weights, key(Role::FfnUpShexp)),
        shared_down: exl3(&weights, key(Role::FfnDownShexp)),
    };
    let mut split = SplitMoe::new(ctx, vec![trunk], bank, &layout, &chunk).expect("split");

    let union: Vec<u16> = {
        let mut seen = [false; 512];
        let mut union = Vec::new();
        for want in &wants {
            for &id in &want.ids {
                if !seen[id] {
                    seen[id] = true;
                    union.push(id as u16);
                }
            }
        }
        union
    };
    assert!(
        union.len() <= 20,
        "the two rows pick at most 20 distinct experts"
    );
    for run in 0..2 {
        let rig = RigRows::new(&projector, &chunk, &row_inputs).expect("rig");
        let passes = split.bank().state().stats().passes;
        split
            .run_rows(&launch, 0, &rig.bufs().expect("bufs"), &chunk)
            .expect("the chunked block");
        stream.synchronize().expect("sync");
        let tag = format!("run {run}");
        // The admit is not decode traffic and never moves the pass counter; a per-row `step`
        // would. This is the assertion that the union-admit path ran, not the tail.
        assert_eq!(
            split.bank().state().stats().passes,
            passes,
            "{tag}: no per-row bank steps"
        );
        for r in 0..m {
            let got = rig.out_row(&projector, r);
            assert!(
                relative_l2(&floats(&got), &floats_f64(&residents[r])) < SPLIT_BOUND,
                "{tag}, row {r}: the all-resident bound"
            );
            assert!(
                relative_l2(&floats(&got), &wants[r].out) < OUTPUT_BOUND,
                "{tag}, row {r}: the fp64 bound"
            );
        }
        for &pick in &union {
            assert!(
                split.bank().state().slot_row(0)[pick as usize] >= 0,
                "{tag}: pick {pick} of the union is in the bank"
            );
        }
    }
}
