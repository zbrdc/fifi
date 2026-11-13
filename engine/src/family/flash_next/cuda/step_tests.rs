//! The device layers of `FlashStep` against the host reference (`hyper.rs`, `qsa.rs`,
//! `deltanet.rs`, `moe.rs`, in the order of `step.rs`) on the real Flash-Next 3.05 bpw pack.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names a local pack directory: a copy with the headers of every shard
//! and the payload of the chosen layers is enough (the sparse layer-3 copy holds layer 3), and
//! `FIFI_FLASH_NEXT_STEP_LAYERS` lists the layers of the window (`3` by default). The full local
//! pack serves any layers.
//!
//! The host reference projects with the AVX-512 CPU kernel on the stored matrices (the exact
//! codebook value, f32 accumulation), which is one binary16 rounding away from the device kernels
//! (about 2e-4 relative per projection, `moe/real_attention_tests.rs`); the routed experts are read
//! and projected on demand. The device keeps the routed experts as an all-resident bank of the
//! window's layers. Each layer is run in steps (hyper read, mixer, apply, hyper read, MoE, apply)
//! with a drain and a read-back after the mixer and after the MoE block, so a divergence names its
//! block. The stream, the block outputs and the status words are compared at every token of a short
//! sequence; the QSA cache and the DeltaNet state therefore have to agree across tokens too.
//!
//! The head check (`FIFI_FLASH_NEXT_STEP_HEAD`) runs the final mixer and `lm_head` on the stream the
//! window leaves, not the full host step: neither side applies the PLE update, so a window that
//! crosses the PLE layer is a truncated model on both sides and its logits only compare the head.
//!
//! Needs a GPU, `--test-threads=1` and (for the full window) several GiB; ignored by default.

use std::{collections::HashMap, path::Path, time::Duration};

use super::{
    bank::{
        BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
        ReplaceOnMissLru, device::DeviceBank,
    },
    bank_source::PackRecordSource,
    load::{
        TrunkList,
        upload::{layer_weights, root_weights},
    },
    moe_block::SplitMoe,
    state::StateSlot,
    step::{AllResidentMoe, FlashStep, MIXER, MoeBlock, SplitMoeAt, bank_key, split_trunk},
    step_plan::{Site, Slot},
};
use crate::{
    EngError,
    family::{
        exl3_pack::{ExpertPart, Root, expert_name_in, layer_name},
        flash_next::{
            deltanet::{self, DeltaState, DeltaWeights},
            hyper::{self, HyperWeights},
            load::model_shape,
            moe::slot_bank::OwnedPart,
            moe::{self, MoeWeights},
            qsa::{self, QsaCache, QsaWeights},
            step::ModelShape,
            stored::host_f32,
        },
        project::{
            DenseProjector, DenseWeight, Projector,
            cuda::{CudaProjector, CudaWeight, CudaWeights, Reserve},
        },
        qwen36::cuda::exl3_source::PackReader,
    },
    kernels::{
        cuda::{DeviceStream, device_present},
        exl3_cpu::{Exl3Isa, Exl3Projection, ExpertCpu},
        nvfp4_cpu::WaitPolicy,
    },
    weights::exl3::{COMPONENT_SUFFIXES, Checkpoint, Matrix},
};

/// Tokens of the sequence a layer runs.
const TOKENS: usize = 10;
/// Cached positions of the run (a multiple of the indexer ratio).
const POSITIONS: usize = 64;

fn checkpoint() -> Checkpoint {
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    Checkpoint::open(Path::new(&dir)).expect("open the pack")
}

fn window() -> Vec<usize> {
    std::env::var("FIFI_FLASH_NEXT_STEP_LAYERS")
        .unwrap_or_else(|_| "3".into())
        .split(',')
        .map(|layer| layer.trim().parse().expect("a layer number"))
        .collect()
}

fn matrix(checkpoint: &Checkpoint, prefix: &str) -> Matrix {
    let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
    let keys = names.each_ref().map(String::as_str);
    let selected = checkpoint
        .source
        .read_selected(&keys, 1 << 30)
        .unwrap_or_else(|e| panic!("{prefix}: {e:?}"));
    Matrix::load(&selected, prefix, &checkpoint.config).expect("a matrix")
}

// ---------------------------------------------------------------------------------------------
// The host reference.

/// A weight of the host reference.
enum W {
    Dense(DenseWeight),
    /// A stored EXL3 matrix, projected by the CPU kernel.
    Exl3(OwnedPart),
    /// The routed experts of `layer`, one stack per part, read on demand.
    Stack(usize, ExpertPart),
}

struct RealProjector<'c> {
    cpu: ExpertCpu,
    checkpoint: &'c Checkpoint,
    experts: HashMap<(usize, usize, usize), OwnedPart>,
}

impl<'c> RealProjector<'c> {
    fn new(checkpoint: &'c Checkpoint) -> Self {
        assert_eq!(
            Exl3Isa::detect(),
            Exl3Isa::Avx512,
            "the reference needs AVX-512"
        );
        Self {
            cpu: ExpertCpu::new(
                4,
                WaitPolicy::SpinThenPark(Duration::from_micros(500)),
                Some(Exl3Isa::Avx512),
            )
            .expect("the CPU kernel"),
            checkpoint,
            experts: HashMap::new(),
        }
    }

    fn project(&self, part: &OwnedPart, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let projection =
            Exl3Projection::new(part.shape, part.words, &part.trellis, &part.suh, &part.svh)?;
        let mut out = vec![0.0f32; part.shape[1]];
        self.cpu.project(&projection, x, &mut out)?;
        Ok(out)
    }
}

impl Projector for RealProjector<'_> {
    type Weight = W;

    fn linear(&mut self, weight: &W, x: &[f32]) -> Result<Vec<f32>, EngError> {
        match weight {
            W::Dense(dense) => DenseProjector::default().linear(dense, x),
            W::Exl3(part) => self.project(part, x),
            W::Stack(..) => Err(EngError::BadRequest("linear on an expert stack".into())),
        }
    }

    fn expert_linear(&mut self, stack: &W, expert: u32, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let W::Stack(layer, part) = stack else {
            return Err(EngError::BadRequest(
                "expert_linear on a plain matrix".into(),
            ));
        };
        let key = (*layer, expert as usize, *part as usize);
        if !self.experts.contains_key(&key) {
            let name = expert_name_in(Root::Layers, *layer, expert as usize, *part);
            let owned = OwnedPart::from_matrix(&matrix(self.checkpoint, &name))?;
            self.experts.insert(key, owned);
        }
        let part = &self.experts[&key];
        self.project(part, x)
    }

    fn rows(&mut self, _: &W, _: &[u32]) -> Result<Vec<f32>, EngError> {
        Err(EngError::BadRequest("rows are not used by a layer".into()))
    }
}

fn dense(checkpoint: &Checkpoint, name: &str, rows: usize, cols: usize) -> W {
    W::Dense(DenseWeight::matrix(
        rows,
        cols,
        host_f32(&checkpoint.source, name, rows * cols).unwrap_or_else(|e| panic!("{name}: {e:?}")),
    ))
}

fn exl3(checkpoint: &Checkpoint, prefix: &str) -> W {
    W::Exl3(OwnedPart::from_matrix(&matrix(checkpoint, prefix)).expect("a part"))
}

fn vector(checkpoint: &Checkpoint, name: &str, len: usize) -> Vec<f32> {
    host_f32(&checkpoint.source, name, len).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

struct HostSite {
    norm: Vec<f32>,
    down: W,
    up: W,
    inject: W,
}

fn host_site(checkpoint: &Checkpoint, model: &ModelShape, prefix: &str) -> HostSite {
    let (width, rank, branches) = (
        model.hyper.width(),
        model.hyper.rank(),
        model.hyper.branches(),
    );
    HostSite {
        norm: vector(checkpoint, &format!("{prefix}.hc_norm.weight"), width),
        down: dense(
            checkpoint,
            &format!("{prefix}.input_mix_weight_down.weight"),
            rank,
            width,
        ),
        up: dense(
            checkpoint,
            &format!("{prefix}.input_mix_weight_up.weight"),
            width,
            rank,
        ),
        inject: dense(
            checkpoint,
            &format!("{prefix}.block_inject_weight.weight"),
            branches,
            width,
        ),
    }
}

impl HostSite {
    fn weights(&self) -> HyperWeights<'_, W> {
        HyperWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: Some(&self.inject),
        }
    }
}

enum HostMixer {
    Delta {
        qkv: W,
        z: W,
        b: W,
        a: W,
        out: W,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
        state: DeltaState,
    },
    Qsa {
        q: W,
        k: W,
        v: W,
        o: W,
        idx: W,
        norms: [Vec<f32>; 4],
        cache: QsaCache,
    },
}

struct HostLayer {
    layer: usize,
    attn: HostSite,
    mlp: HostSite,
    mixer: HostMixer,
    router: Vec<f32>,
    stacks: [W; 3],
    shared: [W; 3],
    shared_scalar: W,
}

fn host_layer(
    checkpoint: &Checkpoint,
    model: &ModelShape,
    kinds: &[bool],
    layer: usize,
) -> HostLayer {
    let h = model.hyper.hidden();
    let at = |suffix: &str| layer_name(layer, suffix);
    let mixer = if kinds[layer] {
        let dims = model.qsa.dims();
        let norm = |suffix: &str, len: u32| vector(checkpoint, &at(suffix), len as usize);
        HostMixer::Qsa {
            q: exl3(checkpoint, &at("self_attn.q_proj")),
            k: exl3(checkpoint, &at("self_attn.k_proj")),
            v: exl3(checkpoint, &at("self_attn.v_proj")),
            o: exl3(checkpoint, &at("self_attn.o_proj")),
            idx: exl3(checkpoint, &at("self_attn.indexer.index_qk_proj")),
            norms: [
                norm("self_attn.q_norm.weight", dims.head_dim),
                norm("self_attn.k_norm.weight", dims.head_dim),
                norm("self_attn.indexer.q_layernorm.weight", dims.idx_dim),
                norm("self_attn.indexer.k_layernorm.weight", dims.idx_dim),
            ],
            cache: QsaCache::new(),
        }
    } else {
        let d = &model.delta;
        let (heads, conv_dim) = (
            d.value_heads(),
            2 * d.key_heads() * d.key_dim() + d.value_heads() * d.value_dim(),
        );
        HostMixer::Delta {
            qkv: exl3(checkpoint, &at("linear_attn.in_proj_qkv")),
            z: exl3(checkpoint, &at("linear_attn.in_proj_z")),
            b: dense(checkpoint, &at("linear_attn.in_proj_b.weight"), heads, h),
            a: dense(checkpoint, &at("linear_attn.in_proj_a.weight"), heads, h),
            out: exl3(checkpoint, &at("linear_attn.out_proj")),
            conv: vector(checkpoint, &at("linear_attn.conv1d.weight"), conv_dim * 4),
            a_log: vector(checkpoint, &at("linear_attn.A_log"), heads),
            dt_bias: vector(checkpoint, &at("linear_attn.dt_bias"), heads),
            norm: vector(checkpoint, &at("linear_attn.norm.weight"), d.value_dim()),
            state: DeltaState::new(&model.delta),
        }
    };
    let part = |p: ExpertPart| W::Stack(layer, p);
    let shared = |suffix: &str| exl3(checkpoint, &at(&format!("mlp.shared_expert.{suffix}")));
    HostLayer {
        layer,
        attn: host_site(checkpoint, model, &at("attn_hyper_connection")),
        mlp: host_site(checkpoint, model, &at("mlp_hyper_connection")),
        mixer,
        router: vector(checkpoint, &at("mlp.gate.weight"), model.moe.experts() * h),
        stacks: [
            part(ExpertPart::Gate),
            part(ExpertPart::Up),
            part(ExpertPart::Down),
        ],
        shared: [shared("gate_proj"), shared("up_proj"), shared("down_proj")],
        shared_scalar: dense(checkpoint, &at("mlp.shared_expert_gate.weight"), 1, h),
    }
}

/// The host blocks of a layer at one token, in the order of `step.rs`.
struct HostRun {
    mixed_attn: Vec<f32>,
    attn_out: Vec<f32>,
    mixed_mlp: Vec<f32>,
    moe_out: Vec<f32>,
}

fn host_token(
    projector: &mut RealProjector<'_>,
    model: &ModelShape,
    layer: &mut HostLayer,
    pos: u32,
    stream: &mut [f32],
) -> HostRun {
    let read = hyper::read(projector, &model.hyper, &layer.attn.weights(), stream).expect("read");
    let gate = read.gate.clone().expect("gate");
    let attn_out = match &mut layer.mixer {
        HostMixer::Qsa {
            q,
            k,
            v,
            o,
            idx,
            norms,
            cache,
        } => qsa::step(
            projector,
            &model.qsa,
            &QsaWeights {
                q,
                k,
                v,
                o,
                idx,
                q_norm: &norms[0],
                k_norm: &norms[1],
                idx_q_norm: &norms[2],
                idx_k_norm: &norms[3],
            },
            cache,
            pos,
            &read.mixed,
        )
        .expect("qsa"),
        HostMixer::Delta {
            qkv,
            z,
            b,
            a,
            out,
            conv,
            a_log,
            dt_bias,
            norm,
            state,
        } => deltanet::step(
            projector,
            &model.delta,
            &DeltaWeights {
                qkv,
                z,
                b,
                a,
                out,
                conv,
                a_log,
                dt_bias,
                norm,
            },
            state,
            &read.mixed,
        )
        .expect("delta"),
    };
    hyper::apply(&model.hyper, stream, &gate, &attn_out).expect("apply");
    let read_m = hyper::read(projector, &model.hyper, &layer.mlp.weights(), stream).expect("read");
    let gate_m = read_m.gate.clone().expect("gate");
    let moe_out = moe::step(
        projector,
        &model.moe,
        &MoeWeights {
            router: &layer.router,
            gate: &layer.stacks[0],
            up: &layer.stacks[1],
            down: &layer.stacks[2],
            shared_gate: &layer.shared[0],
            shared_up: &layer.shared[1],
            shared_down: &layer.shared[2],
            shared_scalar: &layer.shared_scalar,
        },
        &read_m.mixed,
    )
    .expect("moe");
    hyper::apply(&model.hyper, stream, &gate_m, &moe_out).expect("apply");
    HostRun {
        mixed_attn: read.mixed,
        attn_out,
        mixed_mlp: read_m.mixed,
        moe_out,
    }
}

// ---------------------------------------------------------------------------------------------
// The device side.

/// The window's weights on the device: each layer's trunk tensors and its routed expert banks.
fn device_weights(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    layers: &[usize],
    (with_head, with_banks): (bool, bool),
) -> CudaWeights {
    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("trunk list");
    let reader = PackReader::new(projector, checkpoint, Reserve::default());
    let (model, _) = model_shape(&checkpoint.model_config).expect("shape");
    let experts = model.moe.experts();
    let mut all = Vec::new();
    if with_head {
        all.extend(root_weights(&reader, &list).expect("the mixer and lm_head"));
    }
    for &layer in layers {
        all.extend(layer_weights(&reader, &list, layer).expect("the layer's trunk"));
        for part in ExpertPart::ALL.into_iter().filter(|_| with_banks) {
            let bank = projector
                .upload_exl3_bank(experts, Reserve::default(), |expert| {
                    let name = expert_name_in(Root::Layers, layer, expert, part);
                    Ok(matrix(checkpoint, &name))
                })
                .expect("a routed bank");
            let suffix = match part {
                ExpertPart::Gate => "gate_proj",
                ExpertPart::Up => "up_proj",
                ExpertPart::Down => "down_proj",
            };
            let key = bank_key(layer, suffix);
            all.push(CudaWeight::Exl3Bank { name: key, bank });
        }
    }
    CudaWeights::from_exl3(all).expect("the window's weights")
}

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

/// The largest absolute difference.
fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
}

/// A stream whose four branches differ, a different one per token.
fn input(width: usize, hidden: usize, token: usize) -> Vec<f32> {
    (0..width)
        .map(|i| {
            let branch = i / hidden;
            let raw = (((i + 613 * (token + 1)) * 7919 % 2003) as f32 - 1001.0) / 700.0;
            raw * (1.0 + 0.5 * branch as f32)
        })
        .collect()
}

/// Bounds on the relative L2 of each quantity against the host, each about three to four times the
/// worst value measured over ten tokens of the real pack (layer 3, layers 0 to 2, the head, the
/// split block). The device rounds every EXL3 weight (about 2e-4 per projection) and every
/// activation entering an EXL3 projection to binary16, and the host does not; the quantities that
/// never pass through an EXL3 projection (the hyper read of the attention site, all f32) are
/// therefore held far tighter than the block outputs, and a stream that is mostly the residual
/// tighter than a block. A missing term, a wrong head order or a wrong cache row moves a block by
/// 1e-2 or more.
struct Bounds {
    /// Mixed row of the attention site: measured 2.2e-6.
    mixed_attn: f64,
    /// Mixed row of the MLP site, after the attention block's EXL3 projections: measured 9.1e-5.
    mixed_mlp: f64,
    /// The residual stream after a layer: measured 5.5e-5.
    stream: f64,
    /// A block output, the DeltaNet state and ring, the logits: measured 4.6e-4 at most.
    block: f64,
}

const BOUNDS: Bounds = Bounds {
    mixed_attn: 1e-5,
    mixed_mlp: 3e-4,
    stream: 2e-4,
    block: 1.5e-3,
};

/// The MoE block a run uses.
#[derive(Clone, Copy, Debug)]
enum MoeKind {
    /// Qwen3.6's block over routed experts that are all resident (three banks per layer).
    Resident,
    /// The split block over a bank of `slots` slots per layer, held experts `0..slots` at the start.
    Split { slots: usize },
}

/// Bank layer `i` holds pack layer `layers[i]`.
struct WindowRecords<'a, 'b> {
    inner: &'b PackRecordSource<'a>,
    layers: Vec<usize>,
}

impl RecordSource for WindowRecords<'_, '_> {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        self.inner.record(self.layers[layer], expert, out)
    }
}

enum Block<'w> {
    Resident(AllResidentMoe<'w>),
    Split(Box<SplitMoeAt<'w>>),
}

impl Block<'_> {
    fn as_block(&mut self) -> &mut dyn MoeBlock {
        match self {
            Self::Resident(block) => block,
            Self::Split(block) => block.as_mut(),
        }
    }

    /// Misses the bank has served so far (0 for the resident block).
    fn misses(&self) -> u64 {
        match self {
            Self::Resident(_) => 0,
            Self::Split(block) => block.inner.bank().state().stats().misses,
        }
    }
}

/// The split block over a bank of `slots` slots for each of `layers`.
fn split_block<'w>(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    weights: &'w CudaWeights,
    layers: &[usize],
    slots: usize,
    dims: &crate::family::qwen36::cuda::moe::MoeDims,
) -> SplitMoeAt<'w> {
    let source = PackRecordSource::new(checkpoint).expect("the record source");
    let record = source.layout().record_bytes();
    let geometry = BankGeometry::new(layers.len(), dims.n_expert, dims.top_k, slots, record)
        .expect("geometry");
    let held: Vec<i32> = (0..dims.n_expert as i32)
        .map(|expert| {
            if (expert as usize) < slots {
                expert
            } else {
                -1
            }
        })
        .collect();
    let rows: Vec<Vec<i32>> = layers.iter().map(|_| held.clone()).collect();
    let row_refs: Vec<&[i32]> = rows.iter().map(Vec::as_slice).collect();
    let counts: Vec<Vec<u64>> = layers
        .iter()
        .map(|_| (0..dims.n_expert as u64).rev().collect())
        .collect();
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
    let bank = DeviceBank::build(
        projector.context(),
        state,
        &WindowRecords {
            inner: &source,
            layers: layers.to_vec(),
        },
        &plan,
    )
    .expect("build the bank");
    let trunks = layers
        .iter()
        .map(|&layer| split_trunk(weights, layer).expect("the trunk of the split block"))
        .collect();
    SplitMoeAt::new(
        SplitMoe::new(projector.context(), trunks, bank, source.layout(), dims)
            .expect("the split block"),
        layers,
    )
    .expect("consecutive layers")
}

/// What a run leaves for comparison with another run: per token, the MoE output and the stream of
/// the last layer, and the bank's misses over the token (all layers).
struct RunOut {
    moe_out: Vec<Vec<f32>>,
    streams: Vec<Vec<f32>>,
    misses: Vec<u64>,
    /// Per DeltaNet layer of the window, the device recurrent state followed by the conv ring after
    /// the sequence.
    states: Vec<Vec<f32>>,
}

/// How the layers of a token are driven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drive {
    /// Block by block with a drain and a read-back after each, compared with the host.
    Stepped,
    /// All layers queued by one `run_layers_without_ple` call and one drain, as the decode loop
    /// does; only the stream after the last layer is read.
    Composed,
}

/// One run of a window of layers.
#[derive(Clone, Copy, Debug)]
struct Spec<'l> {
    layers: &'l [usize],
    kind: MoeKind,
    tokens: usize,
    /// Replaces `indexer_budget` of the pack's config (the pack's 2048 selects every block of a
    /// short sequence), host reference and device alike.
    budget: Option<u32>,
    drive: Drive,
}

impl<'l> Spec<'l> {
    fn new(layers: &'l [usize], kind: MoeKind) -> Self {
        Self {
            layers,
            kind,
            tokens: TOKENS,
            budget: None,
            drive: Drive::Stepped,
        }
    }
}

/// The pack's config with `indexer_budget` replaced.
fn config_with_budget(config: &[u8], budget: u32) -> Vec<u8> {
    let text = String::from_utf8(config.to_vec()).expect("a utf-8 config");
    let key = "\"indexer_budget\"";
    let at = text.find(key).expect("indexer_budget in the config") + key.len();
    let rest = &text[at..];
    let digits_from = rest.find(|c: char| c.is_ascii_digit()).expect("a number");
    let digits = rest[digits_from..]
        .find(|c: char| !c.is_ascii_digit())
        .expect("the end of the number");
    let (start, end) = (at + digits_from, at + digits_from + digits);
    format!("{}{budget}{}", &text[..start], &text[end..]).into_bytes()
}

fn run_window(layers: &[usize], kind: MoeKind) -> RunOut {
    run_spec(Spec::new(layers, kind))
}

fn run_spec(spec: Spec<'_>) -> RunOut {
    let Spec {
        layers,
        kind,
        tokens,
        budget,
        drive,
    } = spec;
    assert!(
        device_present("Flash-Next layer parity"),
        "a GPU is required"
    );
    let checkpoint = checkpoint();
    let config = match budget {
        Some(budget) => config_with_budget(&checkpoint.model_config, budget),
        None => checkpoint.model_config.clone(),
    };
    let (model, kinds) = model_shape(&config).expect("shape");
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let with_head = std::env::var_os("FIFI_FLASH_NEXT_STEP_HEAD").is_some();
    let weights = device_weights(
        &projector,
        &checkpoint,
        layers,
        (with_head, matches!(kind, MoeKind::Resident)),
    );
    let first = *layers.first().expect("a layer");
    let end = layers.last().expect("a layer") + 1;
    let step = FlashStep::new(ctx, &stream, &weights, &config, POSITIONS, 1, first..end)
        .expect("the step");
    let hidden = model.hyper.hidden();
    let width = model.hyper.width();
    let mut host_projector = RealProjector::new(&checkpoint);
    let mut host: Vec<HostLayer> = layers
        .iter()
        .map(|&layer| host_layer(&checkpoint, &model, &kinds, layer))
        .collect();
    let host_head = with_head.then(|| {
        let width = model.hyper.width();
        let rank = model.hyper.rank();
        (
            vector(&checkpoint, &format!("{MIXER}.hc_norm.weight"), width),
            dense(
                &checkpoint,
                &format!("{MIXER}.input_mix_weight_down.weight"),
                rank,
                width,
            ),
            dense(
                &checkpoint,
                &format!("{MIXER}.input_mix_weight_up.weight"),
                width,
                rank,
            ),
            exl3(&checkpoint, "lm_head"),
        )
    });
    let (words, words_at) = step.words_buffer().expect("words");
    let mut block = match kind {
        MoeKind::Resident => Block::Resident(AllResidentMoe { weights: &weights }),
        MoeKind::Split { slots } => Block::Split(Box::new(split_block(
            &projector,
            &checkpoint,
            &weights,
            layers,
            slots,
            &step.moe_dims(1),
        ))),
    };
    let mut out = RunOut {
        moe_out: Vec::new(),
        streams: Vec::new(),
        misses: Vec::new(),
        states: Vec::new(),
    };

    for token in 0..tokens {
        let mut host_stream = input(width, hidden, token);
        let misses_start = block.misses();
        let mut last = (Vec::new(), Vec::new());
        let stream_range = step.slot(Slot::Stream).expect("stream");
        step.begin_token().expect("begin");
        let params = crate::kernels::cuda::resident::step_params::StepParams::new(1, token as i32)
            .expect("params");
        words
            .write_at(&stream, words_at, &params.to_bytes())
            .expect("params");
        stream_range
            .buffer()
            .write_at(
                &stream,
                stream_range.offset(),
                &host_stream
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .expect("stream");
        if drive == Drive::Composed {
            step.run_layers_without_ple(token, 1, block.as_block())
                .expect("the layers");
            step.finish_token().expect("a clean token");
            let device_stream = step
                .read_f32(step.slot(Slot::Stream).unwrap(), width)
                .expect("stream");
            // Only the stream survives the whole window: the block outputs are lifetime-aliased.
            out.moe_out.push(Vec::new());
            out.streams.push(device_stream);
            out.misses.push(block.misses() - misses_start);
            continue;
        }
        for (index, &layer) in layers.iter().enumerate() {
            let reference = host_token(
                &mut host_projector,
                &model,
                &mut host[index],
                token as u32,
                &mut host_stream,
            );
            // Device, in steps with a read-back after each block.
            let attn_prefix = layer_name(layer, "attn_hyper_connection");
            step.hyper_read(Site::Attn, &attn_prefix, 1)
                .expect("hyper read");
            let mixed = step
                .read_f32(step.slot(Slot::Mixed(Site::Attn)).unwrap(), hidden)
                .expect("mixed");
            step.mixer(layer, token, 1).expect("mixer");
            let attn_out = step
                .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), hidden)
                .expect("block out");
            step.hyper_apply(Site::Attn, 1).expect("apply");
            step.hyper_read(Site::Mlp, &layer_name(layer, "mlp_hyper_connection"), 1)
                .expect("hyper read");
            let mixed_mlp = step
                .read_f32(step.slot(Slot::Mixed(Site::Mlp)).unwrap(), hidden)
                .expect("mixed");
            step.moe_block(layer, block.as_block(), 1).expect("moe");
            let moe_out = step
                .read_f32(step.slot(Slot::BlockOut(Site::Mlp)).unwrap(), hidden)
                .expect("moe out");
            step.hyper_apply(Site::Mlp, 1).expect("apply");
            let device_stream = step
                .read_f32(step.slot(Slot::Stream).unwrap(), width)
                .expect("stream");
            let (status, flag) = step.health().expect("health");
            let pairs: [(&str, &[f32], &[f32], f64); 5] = [
                (
                    "mixed(attn)",
                    &mixed,
                    &reference.mixed_attn,
                    BOUNDS.mixed_attn,
                ),
                ("attn out", &attn_out, &reference.attn_out, BOUNDS.block),
                (
                    "mixed(mlp)",
                    &mixed_mlp,
                    &reference.mixed_mlp,
                    BOUNDS.mixed_mlp,
                ),
                ("moe out", &moe_out, &reference.moe_out, BOUNDS.block),
                ("stream", &device_stream, &host_stream, BOUNDS.stream),
            ];
            let errors: Vec<(&str, f64, f64)> = pairs
                .iter()
                .map(|(name, got, want, bound)| (*name, relative_l2(got, want), *bound))
                .collect();
            println!(
                "layer {layer} token {token}: status {status} flag {flag}; {}",
                pairs
                    .iter()
                    .map(|(name, got, want, _)| format!(
                        "{name} rel {:.2e} max {:.2e} (ref max {:.2e})",
                        relative_l2(got, want),
                        max_abs(got, want),
                        want.iter().fold(0.0f32, |m, v| m.max(v.abs()))
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
            assert_eq!(
                (status, flag),
                (0, 0),
                "layer {layer} token {token}: health"
            );
            for (name, error, bound) in errors {
                assert!(
                    error <= bound,
                    "layer {layer} token {token}: {name} differs from the host by {error:.3e} \
                     (bound {bound:.1e})"
                );
            }
            // The next layer reads the host stream the device holds: keep them in step, so one
            // layer's error does not compound into the next layer's comparison.
            last = (moe_out.clone(), device_stream.clone());
            host_stream = device_stream;
        }
        out.moe_out.push(last.0);
        out.streams.push(last.1);
        out.misses.push(block.misses() - misses_start);
        if host_head.is_none() {
            step.finish_token().expect("a clean token");
        }
        if let Some((norm, down, up, lm_head)) = &host_head {
            step.head(1).expect("the head");
            step.finish_token().expect("a clean token with the head");
            let vocab = step.geometry.vocab;
            let device_logits = step
                .read_f32(step.slot(Slot::Logits).unwrap(), vocab)
                .expect("logits");
            let read = hyper::read(
                &mut host_projector,
                &model.hyper,
                &HyperWeights {
                    norm,
                    down,
                    up,
                    inject: None,
                },
                &host_stream,
            )
            .expect("final mixer");
            let host_logits = host_projector
                .linear(lm_head, &read.mixed)
                .expect("lm_head");
            let top = |logits: &[f32], count: usize| -> Vec<usize> {
                let mut order: Vec<usize> = (0..logits.len()).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                order.truncate(count);
                order
            };
            let (device_top, host_top) = (top(&device_logits, 20), top(&host_logits, 20));
            let overlap = device_top.iter().filter(|id| host_top.contains(id)).count();
            let error = relative_l2(&device_logits, &host_logits);
            println!(
                "token {token}: logits rel {error:.2e} max {:.2e} (ref max {:.2e}); top-1 device {} \
                 host {}; top-20 overlap {overlap}",
                max_abs(&device_logits, &host_logits),
                host_logits.iter().fold(0.0f32, |m, v| m.max(v.abs())),
                device_top[0],
                host_top[0],
            );
            // A truncated model has no preferred token; the top-1 agreement is reported, the
            // logits error is the gate.
            assert!(
                error <= BOUNDS.block,
                "token {token}: logits differ by {error:.3e}"
            );
        }
    }
    // The recurrent state and the conv ring of every DeltaNet layer after the sequence: drift here
    // compounds over a long decode, where a block's error does not.
    for (index, &layer) in layers.iter().enumerate() {
        let HostMixer::Delta { state, .. } = &host[index].mixer else {
            continue;
        };
        let delta = step.geometry.delta;
        step.stream.synchronize().expect("drain");
        let rec = step
            .read_f32(
                step.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                state.rec().len(),
            )
            .expect("recurrent state");
        let ring_device = step
            .read_f32(
                step.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                state.ring().len(),
            )
            .expect("conv ring");
        // The device ring is planar `[tap][channel]` (oldest tap first), the host's `[channel][tap]`.
        let taps = state.ring().len() / delta.conv_dim;
        let mut ring_host_planar = vec![0.0f32; state.ring().len()];
        for channel in 0..delta.conv_dim {
            for tap in 0..taps {
                ring_host_planar[tap * delta.conv_dim + channel] =
                    state.ring()[channel * taps + tap];
            }
        }
        out.states
            .push(rec.iter().chain(&ring_device).copied().collect());
        if drive == Drive::Composed {
            // No host reference ran: the state is compared with the stepped run's, bit for bit.
            continue;
        }
        let rec_error = relative_l2(&rec, state.rec());
        let ring_error = relative_l2(&ring_device, &ring_host_planar);
        println!(
            "layer {layer} after {tokens} tokens: recurrent state rel {rec_error:.2e} max {:.2e} \
             (ref max {:.2e}); conv ring rel {ring_error:.2e} max {:.2e} (ref max {:.2e})",
            max_abs(&rec, state.rec()),
            state.rec().iter().fold(0.0f32, |m, v| m.max(v.abs())),
            max_abs(&ring_device, &ring_host_planar),
            ring_host_planar.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        );
        assert!(
            rec_error <= BOUNDS.block,
            "layer {layer}: recurrent state {rec_error:.3e}"
        );
        assert!(
            ring_error <= BOUNDS.block,
            "layer {layer}: conv ring {ring_error:.3e}"
        );
    }
    out
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_device_layers_match_the_host_reference_on_the_real_pack() {
    run_window(&window(), MoeKind::Resident);
}

/// The indexer's selection has to pick: with the pack's budget of 2048 positions every block of a
/// short sequence is selected and a wrong ranking cannot show. A budget of 8 positions is two
/// blocks of the pack's four; from the 13th token on there are more complete blocks than the
/// selection takes, so the cache, the indexer scores and the selection all decide the output.
/// Run on the window's QSA layers (the DeltaNet layers do not read the budget). fp16 score
/// rounding can flip two near-equal blocks; a flip shows as an error far above the bounds, and is
/// then reported rather than bounded away.
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_selection_of_blocks_matches_the_host_when_it_has_to_choose() {
    let layers = window();
    let (model, kinds) = model_shape(&checkpoint().model_config).expect("shape");
    assert!(
        layers.iter().any(|&layer| kinds[layer]),
        "the window {layers:?} has no QSA layer"
    );
    let tokens = 16;
    let budget = 8;
    let ratio = model.qsa.dims().ratio as usize;
    assert!(
        tokens / ratio > budget / ratio,
        "{tokens} tokens give {} complete blocks, a budget of {budget} takes {}: the selection would not choose",
        tokens / ratio,
        budget / ratio
    );
    run_spec(Spec {
        tokens,
        budget: Some(budget as u32),
        ..Spec::new(&layers, MoeKind::Resident)
    });
}

/// The decode loop queues every layer of a token and drains once. That path has to give the same
/// stream, bit for bit, as the block-by-block run the host comparison uses (same kernels in the
/// same order on the same stream; the drains between blocks must change nothing).
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn the_composed_layers_equal_the_stepped_layers_bit_for_bit() {
    let layers = window();
    let stepped = run_window(&layers, MoeKind::Resident);
    let composed = run_spec(Spec {
        drive: Drive::Composed,
        ..Spec::new(&layers, MoeKind::Resident)
    });
    assert_eq!(stepped.streams.len(), composed.streams.len());
    assert!(
        !stepped.states.is_empty(),
        "the window {layers:?} has no DeltaNet layer: the state comparison would pass vacuously \
         (use FIFI_FLASH_NEXT_STEP_LAYERS=0,1,2,3)"
    );
    assert_eq!(stepped.states.len(), composed.states.len());
    for (index, (a, b)) in stepped.states.iter().zip(&composed.states).enumerate() {
        let equal = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        println!("DeltaNet layer {index} of the window: state and ring bit for bit: {equal}");
        assert!(
            equal,
            "DeltaNet layer {index} of the window: the composed state differs"
        );
    }
    for token in 0..stepped.streams.len() {
        let equal = stepped.streams[token]
            .iter()
            .zip(&composed.streams[token])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "token {token}: composed stream against stepped, rel {:.2e}, bit for bit: {equal}",
            relative_l2(&composed.streams[token], &stepped.streams[token])
        );
        assert!(
            equal,
            "token {token}: the composed run differs from the stepped run"
        );
    }
}

/// A failed call poisons the step: every later call refuses until `reset`, a nonzero status word
/// poisons at `finish_token`, and a `run_layers` that would cross the PLE layer refuses and names
/// the numbers.
#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layer 3)"]
fn a_failed_call_poisons_the_step_until_reset() {
    assert!(device_present("Flash-Next poison"), "a GPU is required");
    let checkpoint = checkpoint();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let weights = device_weights(&projector, &checkpoint, &[3], (false, true));
    let step = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        3..4,
    )
    .expect("the step");
    let mut moe = AllResidentMoe { weights: &weights };

    // A layer whose weights were never uploaded fails, and poisons.
    assert!(!step.is_poisoned());
    let error = step
        .run_layer(5, 0, 1, &mut moe)
        .expect_err("layer 5 has no weights");
    println!("layer 5: {error:?}");
    assert!(step.is_poisoned());
    for refused in [
        step.begin_token(),
        step.run_layer(3, 0, 1, &mut moe),
        step.head(1),
    ] {
        let text = format!("{:?}", refused.expect_err("a poisoned step refuses"));
        assert!(text.contains("reset"), "{text}");
    }
    step.reset().expect("reset");
    assert!(!step.is_poisoned());

    // A clean token finishes clean; a nonzero status word fails the finish and poisons.
    step.begin_token().expect("begin");
    step.finish_token().expect("nothing ran, nothing wrong");
    let status = step.status_word().expect("status word");
    status
        .buffer()
        .write_at(&stream, status.offset(), &1u32.to_le_bytes())
        .expect("write the status word");
    let error = step.finish_token().expect_err("a nonzero status word");
    println!("status word 1: {error:?}");
    assert!(step.is_poisoned());
    step.reset().expect("reset");

    // The PLE refusal names the layer and the range, and does not poison (nothing ran).
    let ple = step.ple_layer();
    let crossing = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        (ple - 1)..(ple + 1),
    )
    .expect("a window across the PLE layer");
    let error = crossing
        .run_layers((ple - 1)..(ple + 1), 0, 1, &mut moe)
        .expect_err("a range across the PLE layer");
    let text = format!("{error:?}");
    println!("PLE: {text}");
    assert!(text.contains(&format!("PLE layer {ple}")), "{text}");
    assert!(!crossing.is_poisoned());
    // A range outside the window refuses too, and does not poison.
    let outside = crossing
        .run_layers(ple..(ple + 5), 0, 1, &mut moe)
        .expect_err("a range outside the window");
    let text = format!("{outside:?}");
    println!("outside: {text}");
    assert!(text.contains("not inside"), "{text}");
    assert!(!crossing.is_poisoned());
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the full pack in FIFI_FLASH_NEXT_EXL3_PACK (6 GiB)"]
fn the_split_moe_block_matches_the_resident_block_and_the_host_across_tokens() {
    let layers = window();
    let resident = run_window(&layers, MoeKind::Resident);
    // (a) A bank that holds every expert: no miss, one pass.
    let full = run_window(&layers, MoeKind::Split { slots: 512 });
    assert!(
        full.misses.iter().all(|&m| m == 0),
        "misses {:?}",
        full.misses
    );
    let mut worst = 0.0f64;
    for token in 0..TOKENS {
        let moe = relative_l2(&full.moe_out[token], &resident.moe_out[token]);
        let stream = relative_l2(&full.streams[token], &resident.streams[token]);
        let bits_equal = full.moe_out[token]
            .iter()
            .zip(&resident.moe_out[token])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "512 slots, token {token}: moe out against the resident block rel {moe:.2e} max {:.2e}, \
             stream rel {stream:.2e}, bit for bit: {bits_equal}",
            max_abs(&full.moe_out[token], &resident.moe_out[token])
        );
        worst = worst.max(moe).max(stream);
    }
    assert!(
        worst <= 1e-5,
        "the 512-slot split block differs from the resident block by {worst:.3e}"
    );
    // (b) A small bank: the picks miss at every token, and the bank state carries across tokens.
    let small = run_window(&layers, MoeKind::Split { slots: 32 });
    println!("32 slots: misses per token {:?}", small.misses);
    assert!(
        small.misses.iter().all(|&m| m > 0),
        "a 32-slot bank must miss at every token: {:?}",
        small.misses
    );
    // Top-10 of 512 experts per layer and token with 32 held: about 9 of the 10 picks miss, so a
    // floor of 3 misses per layer and token leaves room for a lucky token and still fails a bank
    // that does not replace.
    let floor = 3 * layers.len() as u64;
    assert!(
        small.misses.iter().all(|&m| m >= floor),
        "a 32-slot bank must miss at least {floor} times per token: {:?}",
        small.misses
    );
}

#[test]
#[ignore = "needs a GPU, --test-threads=1 and the pack in FIFI_FLASH_NEXT_EXL3_PACK (layers in FIFI_FLASH_NEXT_STEP_LAYERS)"]
fn a_chunk_of_rows_matches_the_same_rows_one_at_a_time() {
    // The chunked prefill contract at the step level: M rows through one rows=M step equal M
    // launches of a rows=1 step, stream rows and DeltaNet states alike. The tolerance covers the
    // EXL3 row-grouped launch's reduction order (the scan itself is bit-exact by contract).
    assert!(
        device_present("Flash-Next chunk parity"),
        "a GPU is required"
    );
    let layers = window();
    let m = 4usize;
    let checkpoint = checkpoint();
    let (model, _kinds) = model_shape(&checkpoint.model_config).expect("shape");
    let hidden = model.hyper.hidden();
    let width = model.hyper.width();
    let projector = CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let stream: DeviceStream = ctx.stream().expect("stream");
    let weights = device_weights(&projector, &checkpoint, &layers, (false, true));
    let first = *layers.first().expect("a layer");
    let end = *layers.last().expect("a layer") + 1;
    let single = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        1,
        first..end,
    )
    .expect("the single-row step");
    let chunk = FlashStep::new(
        ctx,
        &stream,
        &weights,
        &checkpoint.model_config,
        POSITIONS,
        m,
        first..end,
    )
    .expect("the chunk step");

    // One row at a time.
    let mut moe = AllResidentMoe { weights: &weights };
    let mut singles = Vec::new();
    for token in 0..m {
        let row = input(width, hidden, token);
        single.begin_token().expect("begin");
        write_params(&single, 1, token as i32);
        write_stream_row(&single, &row);
        single
            .run_layers_without_ple(token, 1, &mut moe)
            .expect("the layers");
        single.finish_token().expect("a clean token");
        singles.push(
            single
                .read_f32(single.slot(Slot::Stream).unwrap(), width)
                .expect("stream"),
        );
    }

    // The same rows as one chunk.
    let rows: Vec<f32> = (0..m)
        .flat_map(|token| input(width, hidden, token))
        .collect();
    chunk.begin_chunk(m).expect("begin");
    write_params(&chunk, 1, 0);
    write_stream_row(&chunk, &rows);
    chunk
        .run_layers_without_ple(0, m, &mut moe)
        .expect("the layers");
    chunk.finish_token().expect("a clean chunk");
    let chunked = chunk
        .read_f32(chunk.slot(Slot::Stream).unwrap(), m * width)
        .expect("stream");

    for token in 0..m {
        let got = &chunked[token * width..(token + 1) * width];
        let error = relative_l2(got, &singles[token]);
        println!("row {token}: relative L2 {error:.3e}");
        assert!(
            error < 1e-4,
            "row {token} of the chunk drifts from the single-row run: relative L2 {error:.3e}"
        );
    }

    // The DeltaNet recurrent state and conv ring after the sequence.
    for &layer in &layers {
        if chunk.geometry.layers.get(layer) != Some(&crate::family::AttentionKind::DeltaNet) {
            continue;
        }
        let ring_len = single
            .state_slot(StateSlot::DeltaConv(layer))
            .unwrap()
            .bytes()
            / 4;
        let rec_len = single
            .state_slot(StateSlot::DeltaRec(layer))
            .unwrap()
            .bytes()
            / 4;
        single.stream.synchronize().expect("drain");
        let a = single
            .read_f32(
                single.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                rec_len,
            )
            .expect("recurrent state");
        let b = chunk
            .read_f32(
                chunk.state_slot(StateSlot::DeltaRec(layer)).unwrap(),
                rec_len,
            )
            .expect("recurrent state");
        let error = relative_l2(&b, &a);
        println!(
            "layer {layer} recurrent state: relative L2 {error:.3e} ({} values)",
            rec_len
        );
        // The chunk's projections run the job-table kernel (FP16-rounded operands, F32
        // accumulation) where the single rows run legacy F32, and that rounding difference
        // compounds through the recurrent state; 1e-3 covers it with room.
        assert!(
            error < 1e-3,
            "layer {layer}: the chunk's recurrent state drifts beyond the jobs-kernel rounding"
        );
        let ra = single
            .read_f32(
                single.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                ring_len,
            )
            .expect("conv ring");
        let rb = chunk
            .read_f32(
                chunk.state_slot(StateSlot::DeltaConv(layer)).unwrap(),
                ring_len,
            )
            .expect("conv ring");
        let error = relative_l2(&rb, &ra);
        println!("layer {layer} conv ring: relative L2 {error:.3e}");
        assert!(
            error < 1e-3,
            "layer {layer}: the chunk's conv ring drifts beyond the jobs-kernel rounding"
        );
    }
}

fn write_params(step: &FlashStep<'_>, token: i32, pos: i32) {
    let (words, words_at) = step.words_buffer().expect("words");
    let params =
        crate::kernels::cuda::resident::step_params::StepParams::new(token, pos).expect("params");
    words
        .write_at(step.stream, words_at, &params.to_bytes())
        .expect("params");
}

fn write_stream_row(step: &FlashStep<'_>, values: &[f32]) {
    let range = step.slot(Slot::Stream).expect("stream");
    range
        .buffer()
        .write_at(
            step.stream,
            range.offset(),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("stream");
}
