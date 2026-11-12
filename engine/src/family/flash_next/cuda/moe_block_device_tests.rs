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
//!
//! The prefill ring has its own tests: `a_chunk_over_the_prefill_ring_matches_the_all_resident_block`
//! (49 rows over a funded ring, RING_BOUND against the all-resident block and fp64) and
//! [`each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row`] (the launch-family
//! A/B behind that bound). `FIFI_FLASH_NEXT_RING_HELD` (test-only, read by the ring test) picks how
//! many of ROW 0's picks the bank holds — 0 the all-ring arm, 10 the all-bank arm, 6 the default
//! mix — rows 1.. keep the default split in every arm.

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
                relative_l2_f64, upload,
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
/// The ring branch's per-row output deviates from the all-resident block and from fp64 by
/// 1.5e-4 to 2.9e-4 relative L2, measured on this pack at rows=49; the deviation is
/// identical with the fused pair launch on and off and with all, none or some of row 0's
/// picks held in the bank (FIFI_FLASH_NEXT_RING_HELD arms those variants; the knob moves
/// row 0 only — the fixture's other rows keep the default split), so sum order and that
/// source split are excluded. The per-projection device test
/// ([`each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row`])
/// separates the families' arithmetic: the job-table tile kernels stage the activations as
/// F16 operands (`exl3.cu`'s
/// jobs_input_rotate rounds to F16 where it writes the rotated rows; JobsTile multiplies
/// F16 halves) while the flat indexed launches read their F32 staging — giving 1.9-3.9e-4
/// per projection against fp64 where the indexed family sits at 1.5-3.0e-7 — the block
/// deviation is that launch-family arithmetic, not a defect of the ring branch.
/// The bound covers the block-level combine (1.4x headroom over the 2.9e-4 maximum); the
/// per-projection figures are bounded separately at `GATE_UP_BOUND`/`DOWN_BOUND`.
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

/// The buffers of one `run_rows` call, zeroed, with `rows` as the normalized inputs. A
/// source's gate/up launch stages one `xh` row per pair and the whole chunk can be one
/// source's, so the scratch staging is sized `m·k` rows (the F32 byte sizing also covers
/// the F16 staging the job-table family writes into the same ranges).
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

    /// `expert_gu`, `expert_down` (whole chunk) and the routed ids, read back.
    fn expert_bytes(&self, projector: &CudaProjector) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let stream = projector.context().stream().expect("stream");
        let read = |i: usize| {
            let mut bytes = vec![0u8; self.owned[i].bytes()];
            self.owned[i]
                .read_at(&stream, 0, &mut bytes)
                .expect("read the expert buffers");
            bytes
        };
        (read(8), read(10), read(4))
    }
}

/// `run_rows` with the prefill ring funded: every pick reads its bank slot or its ring slot,
/// the bank's miss counter stays flat (the ring path never steps or admits the bank), and
/// every row matches the all-resident block and fp64 within RING_BOUND run after run (the
/// ring re-arms per chunk like the driver cadence). RING_BOUND, not SPLIT_BOUND, because the
/// ring branch's job-table tile launches are not the flat indexed launches of the reference
/// — see RING_BOUND's doc and the per-projection device test that separates the two
/// families' arithmetic.
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

    // The bank holds the first `held_of` of row 0's picks per layer, padded with other
    // experts: the default 6 leaves row 0 mixed; the ring-tolerance measurements hold every
    // pick of row 0 (FIFI_FLASH_NEXT_RING_HELD=10, the all-bank arm) or none (0, the
    // all-ring arm). Rows 1.. keep the default split in every arm.
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

/// The job-table family's per-projection bounds against fp64, split by projection and
/// measured on this pack (Qwen3.8-Flash-Next exl3 3.05bpw h5 ng5, pack layer 3 — the
/// `FIFI_FLASH_NEXT_EXL3_MOE_LAYER` default — its ten picks of one deterministic row, seed 9;
/// 3090): gate/up 1.9-2.2e-4 and down 2.9-3.9e-4, the F16 activation operand class
/// (2^-11 = 4.88e-4; the down bound keeps 1.3x headroom over its 3.9e-4 maximum).
const GATE_UP_BOUND: f64 = 3e-4;
const DOWN_BOUND: f64 = 5e-4;
/// The flat indexed family reads F32 operands: 1.5-3.0e-7 against fp64 on the same row
/// (same provenance), three orders under the jobs family, so the two cannot be confused.
const INDEXED_BOUND: f64 = 1e-6;

/// The launch families the ring tolerance splits between, on one row and one expert each:
/// the flat all-resident block runs the indexed launches (F32 operands), the funded ring runs
/// the job-table tile kernels (F16-staged activations, `exl3.cu`'s jobs_input_rotate and
/// JobsTile), and every projection of the same routed expert is asserted across the two
/// families and against fp64, both twin rows each. Measured on the provenance above:
/// indexed 1.5-3.0e-7 vs fp64, jobs 1.9-2.2e-4 (gate/up) and 2.9-3.9e-4 (down), and
/// jobs-vs-indexed equals jobs-vs-fp64 because the indexed family sits at 1e-7 — the ring
/// branch's whole-block deviation is the job-table family's F16 operand arithmetic, not a
/// defect of the ring branch. The dtype statement rests on the kernel source (the staging
/// write at `exl3.cu`'s jobs_input_rotate rounds to F16; the indexed launches read F32) and
/// on those asserted error classes; no byte-level staging comparison is made. The Qwen3.6
/// chunk prefill test runs these same job-table kernels; its measured print (layer 3, 1024
/// rows, the Qwen3.6 4.00 bpw pack, the host's RTX 3090) is `whole row 8.183e-6; moe term
/// vs residual norm 2.450e-2; delta only 3.341e-4` — the MoE term is 2.45% of the residual
/// stream's norm, and the delta-only figure sits in the 1.9-3.9e-4 job-table class above:
/// the whole-row figure is consistent with that F16-operand error diluted by the residual
/// stream. Margins: down 1.28x over 3.9e-4, gate/up 1.36x over 2.2e-4, indexed 3.4x over
/// 2.9e-7.
#[test]
#[ignore = "needs a GPU (about 3 GiB of ring and bank, about 2 GiB of pinned host memory per bank and FIFI_FLASH_NEXT_EXL3_PACK"]
fn each_projection_through_the_jobs_family_and_the_indexed_family_on_one_row() {
    assert!(
        device_present("Flash-Next split MoE block"),
        "a GPU is required"
    );
    let first = moe_layer();
    let layers = vec![first];
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
    // The split block takes 2 to 64 rows: the row is run twice, identically, so every
    // pick's projection appears once per twin row in the jobs segment; the scan below
    // asserts both twins, so a defect in one twin alone fails.
    let two_rows = MoeDims { rows: 2, ..one_row };
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream = ctx.stream().expect("stream");
    let launch = Launch {
        ctx,
        stream: &stream,
    };
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let layout = source.layout().clone();

    // The flat arm: the all-resident block, one row, indexed launches for every pick.
    let router = Router::read(checkpoint, first, &one_row).expect("router");
    let weights = upload(&projector, checkpoint, (first, &router), &one_row).expect("upload");
    let keys = keys(first);
    let row = input(one_row.hidden, 9);
    let want = reference(checkpoint, (first, &router), &one_row, &row).expect("fp64 reference");
    let block = Exl3MoeWeights::resolve_by(&weights, first, &one_row, |role| {
        keys.iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, key)| key.as_str())
            .ok_or_else(|| EngError::BadRequest(format!("no key for {role:?}")))
    })
    .expect("the all-resident block resolves");
    let flat_rig = RigRows::new(&projector, &two_rows, &[row.clone(), row.clone()]).expect("rig");
    // The flat qwen36 block is per-row: one call computes row 0's picks and fills the gu/down
    // slots at offset 0 (`norm_row` is that row's h bytes).
    run_exl3_moe(
        &launch,
        &block,
        &flat_rig.bufs().expect("bufs"),
        &one_row,
        true,
        None,
        false,
    )
    .expect("the all-resident block");
    launch.stream.synchronize().expect("sync");
    let (flat_gu, flat_down, flat_ids) = flat_rig.expert_bytes(&projector);

    // The jobs arm: a bank with none of the row's picks resident, the ring funded, the same
    // row through the job-table tile kernels.
    let held = others(&want.ids, SLOTS);
    let bank = build_bank(&projector, &source, &layers, &[held]);
    let trunk = {
        let key = |role: Role| {
            keys.iter()
                .find(|(candidate, _)| *candidate == role)
                .map(|(_, key)| key.as_str())
                .unwrap_or_else(|| panic!("no key for {role:?}"))
        };
        SplitMoeTrunk {
            router: raw(&weights, key(Role::FfnGateInp)),
            router_shared: raw(&weights, key(Role::FfnGateInpShexp)),
            shared_gate: exl3(&weights, key(Role::FfnGateShexp)),
            shared_up: exl3(&weights, key(Role::FfnUpShexp)),
            shared_down: exl3(&weights, key(Role::FfnDownShexp)),
        }
    };
    let mut split = SplitMoe::new(ctx, vec![trunk], bank, &layout, &two_rows).expect("split");
    let streamed = shape.experts() - SLOTS;
    let slots = split.plan_ring(ctx, streamed).expect("fund the ring");
    assert!(slots >= streamed, "the ring holds one full streamed layer");
    split.begin_prefill();
    split.begin_chunk();
    let jobs_rig = RigRows::new(&projector, &two_rows, &[row.clone(), row]).expect("rig");
    split
        .run_rows(&launch, 0, &jobs_rig.bufs().expect("bufs"), &two_rows)
        .expect("the ring block");
    stream.synchronize().expect("sync");
    let (jobs_gu, jobs_down, jobs_ids) = jobs_rig.expert_bytes(&projector);

    // The flat block writes the routed ids of row 0 only; both arms route the same logits
    // through the same route kernel, so row 0's picks must agree.
    let (k, i, h) = (one_row.top_k, one_row.inter, one_row.hidden);
    assert_eq!(
        &flat_ids[..k * 4],
        &jobs_ids[..k * 4],
        "both arms route to the same picks"
    );
    // No byte-level staging comparison between the arms is made; the families' difference
    // is established by the asserted error classes below. The dtype statement rests on the
    // kernel source (jobs_input_rotate rounds its operands to F16 where it stores the
    // rotated rows; the indexed launches
    // read the F32 staging) and on the per-projection error classes asserted below: the
    // indexed family reproduces fp64 at 1e-7 while the jobs family sits at 1e-4.
    let (flat_gu, flat_down) = (floats_f64(&flat_gu), floats_f64(&flat_down));
    let (jobs_gu, jobs_down) = (floats_f64(&jobs_gu), floats_f64(&jobs_down));
    // The jobs segment's pair order follows the ring's slot order, not the routing order, so
    // each fp64 pick is matched by scanning the segment: every segment row within the
    // family bound IS this pick (every other expert's projection is orders off — no
    // ambiguity at these gaps), and the row is run twice, identically, so the pick appears
    // once per twin row: exactly two rows must match, and a defect in one twin alone fails.
    let twins = |rows: &[f64], want: &[f64], bound: f64| -> Vec<f64> {
        rows.chunks(want.len())
            .map(|got| relative_l2_f64(got, want))
            .filter(|error| *error < bound)
            .collect()
    };
    let max_of = |v: &[f64]| v.iter().copied().fold(f64::MIN, f64::max);
    for p in 0..k {
        let fg = &flat_gu[p * 2 * i..][..2 * i];
        let wg = &want.gu[p * 2 * i..][..2 * i];
        let fd = &flat_down[p * h..][..h];
        let wd = &want.down[p * h..][..h];
        // The indexed family against fp64, asserted at its own F32 class.
        let (gu_indexed, down_indexed) = (relative_l2_f64(fg, wg), relative_l2_f64(fd, wd));
        assert!(
            gu_indexed < INDEXED_BOUND,
            "pick {p}: indexed gate/up vs fp64 {gu_indexed:.3e}"
        );
        assert!(
            down_indexed < INDEXED_BOUND,
            "pick {p}: indexed down vs fp64 {down_indexed:.3e}"
        );
        // The jobs family against fp64 and against the indexed row, both twins asserted.
        let gu_twins = twins(&jobs_gu, wg, GATE_UP_BOUND);
        let down_twins = twins(&jobs_down, wd, DOWN_BOUND);
        assert_eq!(
            gu_twins.len(),
            2,
            "pick {p}: both gate/up twin rows match fp64 (errors {:?})",
            gu_twins
        );
        assert_eq!(
            down_twins.len(),
            2,
            "pick {p}: both down twin rows match fp64 (errors {:?})",
            down_twins
        );
        let jg = twins(&jobs_gu, fg, GATE_UP_BOUND);
        let jd = twins(&jobs_down, fd, DOWN_BOUND);
        assert_eq!(
            jg.len(),
            2,
            "pick {p}: both gate/up twin rows match the indexed row (errors {:?})",
            jg
        );
        assert_eq!(
            jd.len(),
            2,
            "pick {p}: both down twin rows match the indexed row (errors {:?})",
            jd
        );
        println!(
            "pick {p} gate/up: indexed vs fp64 {gu_indexed:.3e}; jobs vs fp64 {:.3e} (both twins); jobs vs indexed {:.3e}; down: indexed vs fp64 {down_indexed:.3e}; jobs vs fp64 {:.3e}; jobs vs indexed {:.3e}",
            max_of(&gu_twins),
            max_of(&jg),
            max_of(&down_twins),
            max_of(&jd),
        );
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
