//! The whole-model EXL3 load against the real pack.
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack directory; `FIFI_EXL3_LOAD_REPORT`, when set, names a new
//! file that receives the measured numbers as JSON. The test loads every layer, checks each
//! tensor's shape through the step's own lookups, compares the device bytes with an independent
//! sum over the pack's headers and the free memory before and after, and then checks the moved
//! value heads and the bank members against the fp64 reference of the pack's original matrices.

use std::{fs, path::Path, time::Instant};

use serde_json::json;

use super::{
    super::exl3_pack::{ExpertPart, Root, expert_name_in, inspect_pack_split, layer_name},
    common_tests::{dims, inventory},
    exl3_load::{BankedModel, load_exl3_banked_dir, load_exl3_dir},
    layer_table::LayerTable,
    mixer_geometry::MixerGeometry,
    mixer_weights::MixerWeights,
    moe::MoeDims,
    pinned::ROTARY_DIM,
    resources::{GlobalNames, GlobalSpecs, STEP_ROWS},
};
use crate::{
    family::{
        AttentionKind, QWEN3_6_35B_A3B,
        gguf_model::Role,
        project::cuda::{
            CudaProjector, CudaWeight, CudaWeights, Reserve, WeightLayout,
            exl3::exl3_member_staging,
        },
        qwen36::host::HostShape,
    },
    kernels::cuda::device_present,
    tensor::Tensor,
    weights::{
        GgufType, WeightSource,
        exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
    },
};

const EMBED: &str = "model.language_model.embed_tokens.weight";

/// Device bytes the loader must hold, summed from the pack headers alone: trellises as stored,
/// the embedding as stored, every other half-width tensor widened to F32, markers, the vision
/// tower and the MTP draft head not at all.
fn expected_device_bytes(dir: &Path, checkpoint: &Checkpoint) -> u64 {
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let mut total = 0;
    for key in index["weight_map"].as_object().unwrap().keys() {
        let marker = key.ends_with(".mul1") || key.ends_with(".mcg");
        if key.starts_with("mtp.") || key.starts_with("model.visual.") || marker {
            continue;
        }
        let (dtype, shape) = checkpoint.source.stored(key).unwrap();
        let stored = shape.iter().map(|&d| u64::from(d)).product::<u64>() * dtype.byte_width();
        total += if key.ends_with(".trellis") || key == EMBED {
            stored
        } else {
            assert_eq!(dtype.byte_width(), 2, "{key}");
            stored * 2
        };
    }
    total
}

pub(super) fn matrix_of(checkpoint: &Checkpoint, prefix: &str) -> Matrix {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint.source.read_selected(&keys, 1 << 30).unwrap();
    Matrix::load(&selected, prefix, &checkpoint.config).unwrap()
}

fn tensor_of(checkpoint: &Checkpoint, name: &str) -> Tensor {
    let (dtype, shape) = checkpoint.source.stored(name).unwrap();
    let bytes = shape.iter().map(|&d| u64::from(d)).product::<u64>() * dtype.byte_width();
    let selected = checkpoint.source.read_selected(&[name], bytes).unwrap();
    selected.tensor(name).unwrap()
}

pub(super) fn values_of(checkpoint: &Checkpoint, name: &str) -> Vec<f32> {
    let tensor = tensor_of(checkpoint, name);
    let count = tensor.shape().iter().map(|&d| d as usize).product();
    crate::family::qwen36::exl3_tensors::widen_f32(&tensor, count, name).unwrap()
}

fn raw_values(projector: &CudaProjector, weights: &CudaWeights, name: &str) -> Vec<f32> {
    let Some(CudaWeight::Raw { ty, buffer, .. }) = weights.get(name) else {
        panic!("{name} is not a raw tensor");
    };
    assert_eq!(*ty, GgufType::F32, "{name}");
    let stream = projector.context().stream().unwrap();
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(&stream, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

pub(super) fn relative_l2(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm) = (0.0f64, 0.0f64);
    for (a, e) in actual.iter().zip(expected) {
        error += (f64::from(*a) - e).powi(2);
        norm += e * e;
    }
    (error / norm).sqrt()
}

fn input_row(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7 + seed) % 29) as f32 - 14.0) / 16.0)
        .collect()
}

fn reference(matrix: &Matrix, input: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.shape()[1]];
    let input: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    matrix.project_fp64_into(&input, 1, &mut out).unwrap();
    out
}

/// The banked load on the real pack: dense weights plus `slots`-member banks on the device
/// (the header-derived figure, exactly), every routed expert of every layer in the host
/// store, and the store's member bytes identical to the staging of the pack's own matrices.
#[test]
#[ignore = "requires the EXL3 pack in FIFI_EXL3_CHECKPOINT and admitted GPU memory"]
fn banked_load_holds_dense_plus_slots_and_stores_every_expert() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let dir = Path::new(&dir);
    assert!(device_present("banked EXL3 load"), "a GPU is required");
    let slots = 8usize;
    let checkpoint = Checkpoint::open(dir).expect("open the pack");
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM).expect("host shape");
    let report = inspect_pack_split(&checkpoint.source, &shape).expect("header walk");
    let routed = report.expert_bytes;
    let dense = report.device_bytes - routed;
    let layers = shape.kinds.len();
    let experts = shape.moe.experts;
    let per_member = routed / layers as u64 / experts as u64;
    let expected_device = dense + per_member * slots as u64 * layers as u64;
    let projector = CudaProjector::new().expect("projector");
    let before = projector.memory_info().unwrap();
    let BankedModel {
        weights,
        store,
        slots: got_slots,
        experts: got_experts,
        routed_bytes,
        dense_bytes,
    } = load_exl3_banked_dir(&projector, dir, slots, Reserve::default()).expect("banked load");
    assert_eq!(got_slots, slots);
    assert_eq!(got_experts, experts);
    assert_eq!(routed_bytes, routed);
    assert_eq!(dense_bytes, dense);
    assert_eq!(weights.device_bytes() as u64, expected_device);
    assert_eq!(store.layers(), layers);
    assert_eq!(store.host_bytes(), routed);
    let after = projector.memory_info().unwrap();
    let used = (before.free - after.free) as u64;
    assert!(
        used >= expected_device,
        "free fell by {used}, below the expected {expected_device}"
    );
    // The store's member bytes are the staging of the pack's own matrices: spot-check layer
    // 0's three parts at members 0, 1 and the last, through the loader's own helpers.
    for (part_index, part) in [ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down]
        .into_iter()
        .enumerate()
    {
        for expert in [0, 1, experts - 1] {
            let matrix = matrix_of(&checkpoint, &expert_name_in(Root::Layers, 0, expert, part));
            let staged = exl3_member_staging(&matrix).expect("staging");
            let record = store.layer(0)[part_index].member(expert);
            let trellis = &record[..staged.trellis.len()];
            let scales = &record[staged.trellis.len()..];
            assert_eq!(trellis, staged.trellis.as_slice());
            assert_eq!(scales, staged.scales.as_slice());
        }
    }
}

fn device_view<'a>(
    weights: &'a CudaWeights,
    name: &str,
) -> crate::family::project::cuda::Exl3MatrixView<'a> {
    match weights.get(name) {
        Some(CudaWeight::Exl3 { weight, .. }) => weight.view().unwrap(),
        other => panic!(
            "{name} is not an EXL3 matrix: {:?}",
            other.map(CudaWeight::name)
        ),
    }
}

/// The size `cudaMalloc` rounds a large allocation up to on this driver.
const ALLOCATION_GRANULARITY: u64 = 2 << 20;
/// Driver bookkeeping beside the weights: page-table pages and the like.
const DRIVER_SLACK: u64 = 32 << 20;

/// Every projected group of the moved matrices must equal the group of the original.
const MOVED_BOUND: f64 = 1e-5;

struct Pack<'a> {
    projector: &'a CudaProjector,
    checkpoint: &'a Checkpoint,
    weights: &'a CudaWeights,
    table: &'a LayerTable,
    heads: Vec<usize>,
}

impl Pack<'_> {
    fn project(&self, name: &str, input: &[f32]) -> Vec<f32> {
        let view = device_view(self.weights, name);
        let (values, flag) =
            crate::family::project::cuda::exl3::project_one_row(self.projector, &view, input)
                .unwrap();
        assert_eq!(flag, 0, "{name}: arithmetic flag");
        values
    }

    /// Output groups of `name` equal the original's groups at `source[group]`.
    fn moved_outputs(&self, layer: usize, role: Role, suffix: &str, source: &[usize]) -> f64 {
        let name = self.table.name(layer, role).unwrap();
        let original = matrix_of(self.checkpoint, &layer_name(layer, suffix));
        let input = input_row(original.shape()[0], layer);
        let expected = reference(&original, &input);
        let actual = self.project(name, &input);
        let mut moved = Vec::with_capacity(expected.len());
        for &group in source {
            moved.extend_from_slice(&expected[group * 128..(group + 1) * 128]);
        }
        relative_l2(&actual, &moved)
    }

    /// The projection with its input groups moved equals the original on the unmoved input.
    fn moved_inputs(&self, layer: usize) -> f64 {
        let name = self.table.name(layer, Role::SsmOut).unwrap();
        let original = matrix_of(self.checkpoint, &layer_name(layer, "linear_attn.out_proj"));
        let tiled = input_row(original.shape()[0], 3);
        let mut grouped = vec![0.0f32; tiled.len()];
        for (t, &h) in self.heads.iter().enumerate() {
            grouped[h * 128..(h + 1) * 128].copy_from_slice(&tiled[t * 128..(t + 1) * 128]);
        }
        relative_l2(&self.project(name, &tiled), &reference(&original, &grouped))
    }

    /// Member `expert` of the bank `role` equals the original expert matrix.
    fn bank_member(&self, layer: usize, role: Role, part: &str, expert: usize) -> f64 {
        let name = self.table.name(layer, role).unwrap();
        let Some(CudaWeight::Exl3Bank { bank, .. }) = self.weights.get(name) else {
            panic!("{name} is not a bank");
        };
        let prefix = layer_name(layer, &format!("mlp.experts.{expert}.{part}"));
        let original = matrix_of(self.checkpoint, &prefix);
        let input = input_row(original.shape()[0], expert);
        let (actual, flag) = crate::family::project::cuda::exl3::project_one_row(
            self.projector,
            &bank.view(expert).unwrap(),
            &input,
        )
        .unwrap();
        assert_eq!(flag, 0, "{prefix}");
        relative_l2(&actual, &reference(&original, &input))
    }
}

/// Raw vectors of a DeltaNet layer: the decay, the conv taps and a gate row, against the pack.
fn check_raw_vectors(pack: &Pack<'_>, layer: usize, geometry: &MixerGeometry) {
    let d = &geometry.delta;
    let name = |role| pack.table.name(layer, role).unwrap();
    let a = raw_values(pack.projector, pack.weights, name(Role::SsmA));
    let log = values_of(pack.checkpoint, &layer_name(layer, "linear_attn.A_log"));
    let bias = raw_values(pack.projector, pack.weights, name(Role::SsmDtBias));
    let step = values_of(pack.checkpoint, &layer_name(layer, "linear_attn.dt_bias"));
    for (t, &h) in pack.heads.iter().enumerate() {
        assert_eq!(a[t], -log[h].exp(), "decay of tiled head {t}");
        assert_eq!(bias[t], step[h], "step bias of tiled head {t}");
    }
    let conv = raw_values(pack.projector, pack.weights, name(Role::SsmConv1d));
    let stored = values_of(
        pack.checkpoint,
        &layer_name(layer, "linear_attn.conv1d.weight"),
    );
    let lead = (d.channels - d.value_heads * d.value_dim) * d.conv_kernel;
    assert_eq!(conv[..lead], stored[..lead], "q and k taps stay");
    let width = d.value_dim * d.conv_kernel;
    for (t, &h) in pack.heads.iter().enumerate() {
        assert_eq!(
            conv[lead + t * width..lead + (t + 1) * width],
            stored[lead + h * width..lead + (h + 1) * width],
            "taps of tiled head {t}"
        );
    }
    let beta = raw_values(pack.projector, pack.weights, name(Role::SsmBeta));
    let stored = values_of(
        pack.checkpoint,
        &layer_name(layer, "linear_attn.in_proj_b.weight"),
    );
    for (t, &h) in pack.heads.iter().enumerate() {
        let k = geometry.hidden;
        assert_eq!(
            beta[t * k..(t + 1) * k],
            stored[h * k..(h + 1) * k],
            "beta row {t}"
        );
    }
}

/// The norm conventions of the pack: the zero-centered RMS norms (the layer norms and the
/// attention Q and K norms) are stored without their one and load with it added; the gated
/// norm of a DeltaNet layer multiplies by its stored weight and loads unchanged. Flipping either
/// convention gives plausible but wrong text, so each is checked on a layer of its kind.
fn check_norm_conventions(pack: &Pack<'_>, delta: usize, full: usize) {
    let loaded = |layer: usize, role, suffix: &str| {
        let name = pack.table.name(layer, role).unwrap();
        let device = raw_values(pack.projector, pack.weights, name);
        let stored = values_of(pack.checkpoint, &layer_name(layer, suffix));
        (device, stored)
    };
    for (layer, role, suffix) in [
        (delta, Role::AttnNorm, "input_layernorm.weight"),
        (
            delta,
            Role::PostAttentionNorm,
            "post_attention_layernorm.weight",
        ),
        (full, Role::AttnQNorm, "self_attn.q_norm.weight"),
        (full, Role::AttnKNorm, "self_attn.k_norm.weight"),
    ] {
        let (device, stored) = loaded(layer, role, suffix);
        assert!(!stored.is_empty(), "{suffix}");
        assert!(
            device.iter().zip(&stored).all(|(d, s)| *d == s + 1.0) && device.len() == stored.len(),
            "layer {layer} {suffix} adds one"
        );
    }
    let (device, stored) = loaded(delta, Role::SsmNorm, "linear_attn.norm.weight");
    assert_eq!(device, stored, "the gated norm loads without the one");
}

/// Every tensor the step reads resolves with the shape the step needs.
fn check_shapes(weights: &CudaWeights, table: &LayerTable, geometry: &MixerGeometry) -> usize {
    let layers: Vec<usize> = (0..table.len()).collect();
    MixerWeights::new(geometry, table, weights, &layers, WeightLayout::Exl3)
        .map(|_| ())
        .expect("every mixer weight resolves");
    let d = dims();
    let moe = MoeDims::new(&d, STEP_ROWS);
    let mut banks = 0;
    for layer in 0..table.len() {
        let name = |role| table.name(layer, role).unwrap();
        for (role, k, n) in [
            (Role::FfnGateExps, moe.hidden, moe.inter),
            (Role::FfnUpExps, moe.hidden, moe.inter),
            (Role::FfnDownExps, moe.inter, moe.hidden),
        ] {
            let Some(CudaWeight::Exl3Bank { bank, .. }) = weights.get(name(role)) else {
                panic!("layer {layer} {role:?} is not a bank");
            };
            assert_eq!(
                (bank.count(), bank.shape()),
                (moe.n_expert, [k, n]),
                "layer {layer} {role:?}"
            );
            banks += 1;
        }
        for (role, k, n) in [
            (Role::FfnGateShexp, moe.hidden, moe.shared_inter),
            (Role::FfnUpShexp, moe.hidden, moe.shared_inter),
            (Role::FfnDownShexp, moe.shared_inter, moe.hidden),
        ] {
            let view = device_view(weights, name(role));
            assert_eq!(view.shape, [k, n], "layer {layer} {role:?}");
        }
        for (role, values) in [
            (Role::FfnGateInp, moe.n_expert * moe.hidden),
            (Role::FfnGateInpShexp, moe.hidden),
        ] {
            let Some(CudaWeight::Raw { buffer, .. }) = weights.get(name(role)) else {
                panic!("layer {layer} {role:?} is not raw");
            };
            assert_eq!(buffer.bytes(), values * 4, "layer {layer} {role:?}");
        }
        if table.slot(layer).unwrap().kind == AttentionKind::DeltaNet {
            for role in [Role::SsmBeta, Role::SsmAlpha] {
                let Some(CudaWeight::Raw { buffer, .. }) = weights.get(name(role)) else {
                    panic!("layer {layer} {role:?} is not raw");
                };
                assert_eq!(
                    buffer.bytes(),
                    32 * moe.hidden * 4,
                    "layer {layer} {role:?}"
                );
            }
        }
    }
    let globals = GlobalNames::new(&inventory()).unwrap();
    let specs = GlobalSpecs::new(&d);
    let Some(CudaWeight::Raw { ty, buffer, .. }) = weights.get(&globals.embed) else {
        panic!("embedding is not a raw table");
    };
    assert_eq!(*ty, GgufType::Bf16);
    assert_eq!(buffer.bytes(), specs.embed.rows * specs.embed.k * 2);
    assert_eq!(
        device_view(weights, &globals.head).shape,
        [specs.head.k, specs.head.rows]
    );
    banks
}

#[test]
#[ignore = "requires admitted GPU memory (17 GiB) and the pinned EXL3 pack in FIFI_EXL3_CHECKPOINT"]
fn real_pack_loads_every_layer_with_the_shapes_and_bytes_of_the_step() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let dir = Path::new(&dir);
    assert!(
        device_present("real EXL3 whole-model load"),
        "a GPU is required"
    );
    let checkpoint = Checkpoint::open(dir).expect("open the pack");
    let expected = expected_device_bytes(dir, &checkpoint);
    drop(checkpoint);
    let projector = CudaProjector::new().expect("projector");
    let before = projector.memory_info().unwrap();
    let began = Instant::now();
    let weights = load_exl3_dir(&projector, dir, Reserve::default()).expect("whole-model load");
    let seconds = began.elapsed().as_secs_f64();
    let after = projector.memory_info().unwrap();
    assert_eq!(weights.layout().unwrap(), WeightLayout::Exl3);
    let held = weights.device_bytes() as u64;
    assert_eq!(held, expected, "device bytes against the header sum");
    // The free memory falls by the weights, each buffer rounded up to the driver's allocation
    // granularity, and nothing else: no kernel runs during a load. The bound is computed from
    // the actual buffer sizes, so it cannot be fitted to one run.
    let used = (before.free - after.free) as u64;
    let buffers = weights.device_buffers().count();
    let rounded: u64 = weights
        .device_buffers()
        .map(|buffer| (buffer.bytes() as u64).next_multiple_of(ALLOCATION_GRANULARITY))
        .sum();
    println!(
        "{buffers} buffers, {held} B held, {rounded} B rounded up, {used} B of free memory used"
    );
    assert!(
        used >= held && used <= rounded + DRIVER_SLACK,
        "free memory fell by {used}; {held} held, at most {rounded} after rounding {buffers} buffers"
    );

    let table = LayerTable::new(&dims(), &inventory()).unwrap();
    let geometry = MixerGeometry::new(&dims()).unwrap();
    let banks = check_shapes(&weights, &table, &geometry);

    let checkpoint = Checkpoint::open(dir).unwrap();
    let heads = crate::family::qwen36::exl3_tensors::tiled_head_order(16, 32).unwrap();
    let pack = Pack {
        projector: &projector,
        checkpoint: &checkpoint,
        weights: &weights,
        table: &table,
        heads: heads.clone(),
    };
    let delta = (0..table.len())
        .find(|&l| table.slot(l).unwrap().kind == AttentionKind::DeltaNet)
        .unwrap();
    let qkv_source: Vec<usize> = (0..32).chain(heads.iter().map(|h| 32 + h)).collect();
    let errors = [
        (
            "z",
            pack.moved_outputs(delta, Role::AttnGate, "linear_attn.in_proj_z", &heads),
        ),
        (
            "qkv",
            pack.moved_outputs(delta, Role::AttnQkv, "linear_attn.in_proj_qkv", &qkv_source),
        ),
        ("out", pack.moved_inputs(delta)),
        (
            "gate expert 17 layer 5",
            pack.bank_member(5, Role::FfnGateExps, "gate_proj", 17),
        ),
        (
            "up expert 100 layer 20",
            pack.bank_member(20, Role::FfnUpExps, "up_proj", 100),
        ),
        (
            "down expert 255 layer 39",
            pack.bank_member(39, Role::FfnDownExps, "down_proj", 255),
        ),
    ];
    for (what, error) in &errors {
        println!("{what}: relative L2 {error:.3e}");
        assert!(*error < MOVED_BOUND, "{what}: {error}");
    }
    check_raw_vectors(&pack, delta, &geometry);
    let full = (0..table.len())
        .find(|&l| table.slot(l).unwrap().kind == AttentionKind::Full)
        .unwrap();
    check_norm_conventions(&pack, delta, full);
    let norm = raw_values(
        &projector,
        &weights,
        &GlobalNames::new(&inventory()).unwrap().norm,
    );
    let stored = values_of(&checkpoint, "model.language_model.norm.weight");
    assert!(
        norm.iter().zip(&stored).all(|(a, s)| *a == s + 1.0),
        "output norm adds one"
    );

    println!(
        "EXL3 load: {seconds:.1} s, {held} B resident, free {} -> {}",
        before.free, after.free
    );
    if let Some(path) = std::env::var_os("FIFI_EXL3_LOAD_REPORT") {
        let report = json!({
            "load_seconds": seconds, "device_bytes": held, "expected_bytes": expected,
            "free_before": before.free, "free_after": after.free, "banks": banks,
            "layers": table.len(), "tensors": weights.len(),
            "relative_l2": errors.iter().map(|(w, e)| json!({"case": w, "error": e})).collect::<Vec<_>>(),
        });
        let text = serde_json::to_vec_pretty(&report).unwrap();
        fs::write(path, text).expect("write the report");
    }
}
