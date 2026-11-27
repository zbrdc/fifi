//! The Raon-1 forward pass on one CUDA device.
//!
//! The stack is composed from the shared GGUF resident kernels (`matvec_q` and `matvec_expert`
//! over the file's own blocks, the F16 `dense_gemv`, `moe_combine`) and the row operations of
//! `kernels::cuda::resident::block_ops` (mixed residual, short convolution, delta-rule scan,
//! latent attention, SiTU, biased router). Every operation works on a chunk of token rows, so
//! prefill and decode share one path: a chunk of `n` tokens runs layer by layer, and decode is
//! the chunk of one. Quantized dense projections take up to 8 activation rows per launch and
//! larger chunks take the batched product; the routed experts run per token up to 8 rows and as
//! grouped products, one per expert stack, above that.
//!
//! The embedding table stays in the file: each token's row is dequantized on the host and
//! uploaded into the four residual streams. The latent keys of the seven attention layers are
//! kept as f16 rows of 576 values; the delta-rule state and the convolution history of the
//! other eighteen layers are f32.

use std::cell::{Cell, RefCell};
use std::path::Path;

use super::reference::{Hyper, hyper};
use super::{RAON_1_MINI_PREVIEW, Raon1Model, Raon1Spec, open_raon_1_mini_preview};
use crate::EngError;
use crate::family::mem::DEVICE;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeight, CudaWeights, LoadOptions, Reserve, WeightLayout, served_admission,
};
use crate::family::step::attend::latent;
use crate::family::step::recipe::{Forward, Layer, Mixer, Model, Site, Stage as WalkStage, Walk};
use crate::family::step::rows::{Map, Rows, StepIo};
use crate::family::step::scratch::{Alloc, Core, Count, OnDevice, Widths, Zeroed};
use crate::family::step::state::{Slots, State};
use crate::family::step::walk::{self, ChunkStep, Half};
use crate::family::step::{delta, ffn, head, moe, site};
use crate::kernels::cuda::resident::block_ops::{BlockOp, MLA_GROUP};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::GgufType;
use crate::weights::dequantize;

/// Token rows per chunk: the scratch is sized for this many.
pub(crate) const CHUNK: usize = 256;

// The geometry of the checkpoint this device path serves, read from its spec (the opener pins
// the same fields against the GGUF header). The kernels of `block_ops` fix the head width, the
// latent rank and the rope width; `geometry_check` refuses a spec outside them by name.
const SPEC: Raon1Spec = RAON_1_MINI_PREVIEW;
const HIDDEN: usize = SPEC.hidden as usize;
const STREAMS: usize = SPEC.streams as usize;
const HEADS: usize = SPEC.heads as usize;
const HEAD_DIM: usize = SPEC.head_dim as usize;
const WIDTH: usize = HEADS * HEAD_DIM;
const RANK: usize = SPEC.kv_latent as usize;
const ROPE: usize = (SPEC.kv_a_width - SPEC.kv_latent) as usize;
const MLA_ROW: usize = RANK + ROPE;
const Q_NOPE: usize = SPEC.k_head as usize;
const Q_ROW: usize = Q_NOPE + ROPE;
const Q_LORA: usize = SPEC.q_lora as usize;
const EXPERTS: usize = SPEC.experts as usize;
const USED: usize = SPEC.experts_used as usize;
const MOE_INTER: usize = SPEC.moe_inter as usize;
const DENSE_INTER: usize = SPEC.dense_inter as usize;
const TIME_RANK: usize = SPEC.time_step_rank as usize;
const CONV_TAPS: usize = SPEC.conv_kernel as usize - 1;
// The row operations fix these widths: a spec outside them does not build.
const _: () = assert!(
    HEAD_DIM == 128
        && RANK == 512
        && ROPE == 64
        && Q_NOPE == 128
        && STREAMS == 4
        && HEADS.is_multiple_of(8)
);

/// The widths of a delta-rule layer.
const DELTA: delta::Dims = delta::Dims {
    key_heads: HEADS,
    key_dim: HEAD_DIM,
    value_heads: HEADS,
    value_dim: HEAD_DIM,
    conv_kernel: SPEC.conv_kernel as usize,
    conv_dim: 3 * WIDTH,
};

/// The widths of a routed block.
const MOE: moe::Dims = moe::Dims {
    hidden: HIDDEN,
    inner: MOE_INTER,
    experts: EXPERTS,
    used: USED,
    gated_shared: false,
};

/// The three convolutions of a delta-rule layer in the order of their columns, and the name
/// their joined taps take in the derived tensors.
const CONVS: [&str; 3] = ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"];
const CONV: &str = "ssm_conv1d.weight";
/// Bytes of KV cache per position: seven attention layers of 576 f16 values.
const KV_BYTES_PER_POSITION: u64 = SPEC.mla_layers.len() as u64 * MLA_ROW as u64 * 2;
/// Smallest context a start accepts.
const MIN_CONTEXT: usize = 2048;
/// The model's trained context.
const MAX_CONTEXT: usize = 262_144;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn plan_error(message: String) -> EngError {
    EngError::MemoryPlan(format!("raon1: {message}"))
}

/// The recipe entry of layer `layer`.
fn layer_recipe(spec: Raon1Spec, hy: &Hyper, layer: usize) -> Layer {
    let latent = spec.mla_layers.contains(&layer);
    let ordinal = (0..layer)
        .filter(|l| spec.mla_layers.contains(l) == latent)
        .count();
    let n = |suffix: &str| format!("blk.{layer}.{suffix}");
    let three =
        |stem: &str, tail: &str| ["gate", "up", "down"].map(|t| n(&format!("{stem}{t}{tail}")));
    let site = |name: &str| ["fn", "base", "scale"].map(|t| n(&format!("hc_{name}_{t}.weight")));
    let mixer = if latent {
        Mixer::Latent {
            slot: ordinal,
            names: Box::new(latent::Names {
                query_low: n("attn_q_a.weight"),
                query_norm: n("attn_q_a_norm.weight"),
                query: n("attn_q_b.weight"),
                latent: n("attn_kv_a_mqa.weight"),
                latent_norm: n("attn_kv_a_norm.weight"),
                gate: n("attn_gate.weight"),
                out: n("attn_output.weight"),
            }),
        }
    } else {
        Mixer::Delta {
            slot: ordinal,
            names: Box::new(delta::DeltaNames {
                rule: delta::RuleNames::Channels {
                    inputs: ["attn_q", "attn_k", "attn_v", "ssm_g"]
                        .map(|t| n(&format!("{t}.weight"))),
                    decay_a: n("ssm_f_a.weight"),
                    decay_b: n("ssm_f_b.weight"),
                    rank: TIME_RANK,
                    beta: n("ssm_beta.weight"),
                    delta: n("ssm_rda.weight"),
                    lower_bound: hy.gate_lower_bound,
                    norm_eps: hy.norm_eps,
                },
                out: n("attn_output.weight"),
                conv: n(CONV),
                a: n("ssm_a"),
                dt_bias: n("ssm_dt.bias"),
                norm: n("ssm_norm.weight"),
            }),
        }
    };
    let forward = if layer >= spec.leading_dense {
        Forward::Routed(moe::Names {
            router: n("ffn_gate_inp.weight"),
            rule: moe::Router::Sigmoid {
                bias: n("exp_probs_b.bias"),
            },
            stacks: three("ffn_", "_exps.weight"),
            shared: three("ffn_", "_shexp.weight"),
        })
    } else {
        Forward::Dense(three("ffn_", ".weight"))
    };
    Layer {
        norms: [n("attn_norm.weight"), n("ffn_norm.weight")].map(Some),
        site: Site::Mixed {
            halves: [site("attn"), site("ffn")],
        },
        mixer,
        forward,
    }
}

/// The per-layer f32 factors of the absorbed attention, dequantized once at load.
struct MlaFactors {
    wk_b: DeviceBuffer,
    wv_b: DeviceBuffer,
}

/// The scratch of one chunk, allocated once at load.
struct Scratch<B = DeviceBuffer> {
    core: Core<B>,
    moe: moe::Scratch<B>,
    /// The mean of the streams of the row the head reads.
    head: B,
    branch: B,
    post: B,
    comb: B,
    scan_decay: B,
    scan_proj: B,
    g1: B,
    g2: B,
    fa: B,
    beta: B,
    delta: B,
    qa: B,
    q8_qa: B,
    qb: B,
    kv: B,
    qeff: B,
    part: B,
    mix: B,
    mla_out: B,
    ffn_gate: B,
    ffn_up: B,
    q8_ffn: B,
    mla_gated: B,
    act_ffn: B,
}

impl Scratch {
    fn ffn_scratch<'a>(
        &'a self,
        n: usize,
        width: usize,
        act: Rows<'a>,
    ) -> Result<ffn::Scratch<'a>, EngError> {
        ffn::Scratch::packed(&self.ffn_gate, &self.ffn_up, (n, width), act)
    }

    fn cur_rows(&self) -> Rows<'_> {
        self.core.normed_rows()
    }
}

pub(crate) struct Raon1Device {
    ctx: CudaContext,
    stream: DeviceStream,
    weights: CudaWeights,
    model: Raon1Model,
    spec: Raon1Spec,
    hy: Hyper,
    layers: Vec<Layer>,
    mla: Vec<MlaFactors>,
    /// The joined taps of the `[q | k | v]` convolution of every delta-rule layer, by name.
    derived: CudaWeights,
    sc: Scratch,
    /// The record and the convolution history of every delta-rule layer.
    state: State,
    /// `[mla layers][cells][576]` f16 latent cache.
    cache: DeviceBuffer,
    cells: usize,
    vocab: usize,
    trace: RefCell<Option<Vec<Stage>>>,
    /// Rows of the chunk being walked, for the trace.
    rows: Cell<usize>,
    /// Every product on the decoding routes over f32 rows: the reference of an accuracy run.
    exact: Cell<bool>,
}

/// One named stage of a traced feed.
pub(crate) type Stage = (String, Vec<f32>);

pub(crate) use crate::family::step::walk::HeadRows;

impl Raon1Device {
    /// Opens the GGUF at `path` strictly and loads it onto the device. `context_limit` caps the
    /// positions the cache holds; `None` takes what the free device memory admits, up to the
    /// model's trained context. `announce` receives the planned context before the weights upload.
    pub(crate) fn load(
        path: &Path,
        context_limit: Option<usize>,
        announce: &mut dyn FnMut(usize),
    ) -> Result<Self, EngError> {
        let model = open_raon_1_mini_preview(path, None)?;
        let spec = model.spec();
        let hy = hyper(&model, spec)?;
        if !hy.weights_norm {
            return Err(EngError::Checkpoint(
                "raon1: the shared router normalizes the selected weights; expert_weights_norm is false in this file"
                    .into(),
            ));
        }
        let vocab = spec.vocab as usize;
        let ctx = CudaContext::with_admission(&served_admission())?;
        let stream = ctx.stream()?;
        let names: Vec<String> = spec
            .inventory()
            .into_iter()
            .map(|expected| expected.name)
            .filter(|name| {
                name != "token_embd.weight"
                    && !name.ends_with("attn_k_b.weight")
                    && !name.ends_with("attn_v_b.weight")
                    && !CONVS
                        .iter()
                        .any(|conv| name.ends_with(&format!("{conv}.weight")))
            })
            .collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let tensors = crate::family::project::cuda::native_load_bytes(model.weights(), &name_refs)?;
        let mla_factor_bytes = (spec.mla_layers.len() * 2 * HEADS * RANK * HEAD_DIM * 4) as u64;
        let fixup = ctx.mmq_fixup_bytes()?;
        let plan = Count::default();
        scratch(&plan, vocab, fixup)?;
        let scratch_bytes = plan.total();
        let fixed =
            tensors + mla_factor_bytes + scratch_bytes + state_bytes(spec)? + DEVICE.raon_headroom;
        let free = u64::try_from(ctx.memory_info()?.free).unwrap_or(u64::MAX);
        let room = free.checked_sub(fixed).ok_or_else(|| {
            plan_error(format!(
                "the weights and scratch need {fixed} bytes with the headroom, the device has \
                 {free} free"
            ))
        })?;
        let affordable = usize::try_from(room / KV_BYTES_PER_POSITION)
            .unwrap_or(usize::MAX)
            .min(MAX_CONTEXT)
            / 256
            * 256;
        let cells = match context_limit {
            Some(wanted) if wanted > affordable => {
                return Err(plan_error(format!(
                    "a context of {wanted} tokens needs {} bytes of cache, the device has room \
                     for {affordable} tokens",
                    wanted as u64 * KV_BYTES_PER_POSITION
                )));
            }
            Some(wanted) => wanted,
            None => affordable,
        };
        if cells < MIN_CONTEXT {
            return Err(plan_error(format!(
                "the device has room for {affordable} tokens of context after the weights, below \
                 the minimum of {MIN_CONTEXT}"
            )));
        }
        announce(cells);
        let reserve = Reserve {
            kv_cache: cells as u64 * KV_BYTES_PER_POSITION,
            work: mla_factor_bytes + scratch_bytes + state_bytes(spec)?,
            headroom: DEVICE.raon_headroom,
        };
        let loader = BorrowedLoader::new(&ctx)?;
        let weights = loader.load_sliced(
            model.weights(),
            &name_refs,
            LoadOptions {
                reserve,
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
            &[],
            0,
        )?;
        let layers: Vec<Layer> = (0..spec.layers)
            .map(|l| layer_recipe(spec, &hy, l))
            .collect();
        let mut mla = Vec::new();
        for &layer in spec.mla_layers {
            let upload = |name: &str| -> Result<DeviceBuffer, EngError> {
                ctx.upload(&f32_bytes(&model.read_f32(&format!("blk.{layer}.{name}"))?))
            };
            mla.push(MlaFactors {
                wk_b: upload("attn_k_b.weight")?,
                wv_b: upload("attn_v_b.weight")?,
            });
        }
        let mut conv = Vec::new();
        for layer in (0..spec.layers).filter(|layer| !spec.mla_layers.contains(layer)) {
            let mut taps = Vec::with_capacity(3 * WIDTH * SPEC.conv_kernel as usize);
            for name in CONVS {
                taps.extend(model.read_f32(&format!("blk.{layer}.{name}.weight"))?);
            }
            conv.push(CudaWeight::Raw {
                name: format!("blk.{layer}.{CONV}"),
                ty: GgufType::F32,
                buffer: ctx.upload(&f32_bytes(&taps))?,
            });
        }
        let derived = CudaWeights::from_weights(conv)?;
        let map = Map {
            family: "raon1",
            weights: &weights,
            derived: Some(&derived),
        };
        layers.iter().try_for_each(|l| l.check(&map, DELTA, None))?;
        let sc = scratch(&OnDevice(&ctx), vocab, fixup)?;
        let state = State::new(&Zeroed(OnDevice(&ctx)), slots(spec, cells))?;
        let cache = ctx.upload(&vec![0u8; spec.mla_layers.len() * cells * MLA_ROW * 2])?;
        Ok(Self {
            ctx,
            stream,
            weights,
            model,
            spec,
            hy,
            layers,
            mla,
            derived,
            sc,
            state,
            cache,
            cells,
            vocab,
            trace: RefCell::new(None),
            rows: Cell::new(0),
            exact: Cell::new(false),
        })
    }

    /// Positions the latent cache holds.
    pub(crate) fn cells(&self) -> usize {
        self.cells
    }

    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    /// Starts a new sequence: clears the delta-rule state and the convolution history. The
    /// latent cache is position-addressed and needs no clearing.
    pub(crate) fn reset(&mut self) -> Result<(), EngError> {
        self.state.reset(&self.stream)?;
        self.stream.synchronize()
    }

    /// Records the named stages of the next feed as f32 vectors (residual, norm and gated mixer
    /// outputs per layer), for comparison with the host reference.
    #[cfg(test)]
    pub(crate) fn trace_next_feed(&self) {
        *self.trace.borrow_mut() = Some(Vec::new());
    }

    #[cfg(test)]
    pub(crate) fn take_trace(&self) -> Vec<Stage> {
        self.trace.borrow_mut().take().unwrap_or_default()
    }

    /// Feeds `tokens` at positions `pos0..`, in chunks. Returns the logits of the rows `head`
    /// names (one row of `vocab` values each).
    pub(crate) fn feed(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        head: HeadRows,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        crate::family::step::walk::feed(self, tokens, pos0, head)
    }

    fn io(&self) -> StepIo<'_> {
        Model::io(self, self.exact.get())
    }

    /// Puts every product on the decoding routes over f32 rows, or back on the served ones.
    #[cfg(test)]
    pub(crate) fn set_exact(&self, on: bool) {
        self.exact.set(on);
    }

    fn snap(&self, name: &str, buffer: &DeviceBuffer, values: usize) -> Result<(), EngError> {
        if self.trace.borrow().is_none() {
            return Ok(());
        }
        let mut bytes = vec![0u8; values * 4];
        buffer.read_at(&self.stream, 0, &mut bytes)?;
        if let Some(trace) = self.trace.borrow_mut().as_mut() {
            trace.push((name.to_owned(), f32_values(&bytes)));
        }
        Ok(())
    }

    /// The kept experts of the `n` rows of layer `layer` join the trace, as floats.
    fn snap_ids(&self, layer: usize, n: usize) -> Result<(), EngError> {
        let mut bytes = vec![0u8; n * USED * 4];
        self.sc.moe.ids.read_at(&self.stream, 0, &mut bytes)?;
        let ids = bytes.as_chunks::<4>().0.iter();
        let ids = ids.map(|c| i32::from_le_bytes(*c) as f32).collect();
        if let Some(trace) = self.trace.borrow_mut().as_mut() {
            trace.push((format!("moe_ids-{layer}"), ids));
        }
        Ok(())
    }

    /// One embedding row, dequantized on the host from the type the file stores it in.
    fn embedding_row(&self, token: u32) -> Result<Vec<f32>, EngError> {
        let name = "token_embd.weight";
        let ty = self
            .model
            .header()
            .tensor(name)
            .ok_or_else(|| EngError::Checkpoint(format!("raon1: tensor {name} is missing")))?
            .ggml_type();
        let row_bytes = (HIDDEN as u64 / ty.block_elems() * ty.block_bytes()) as usize;
        let mut bytes = vec![0u8; row_bytes];
        self.model
            .weights()
            .read_range(name, u64::from(token) * row_bytes as u64, &mut bytes)?;
        dequantize(ty, &bytes)
    }

    fn chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError> {
        let n = tokens.len();
        let mut resid = vec![0f32; n * STREAMS * HIDDEN];
        for (t, &token) in tokens.iter().enumerate() {
            let row = self.embedding_row(token)?;
            for s in 0..STREAMS {
                let at = (t * STREAMS + s) * HIDDEN;
                resid[at..at + HIDDEN].copy_from_slice(&row);
            }
        }
        self.sc.core.resid[0].write_at(&self.stream, 0, &f32_bytes(&resid))?;
        self.rows.set(n);
        walk::visits(&Walk(&*self), n, pos0, |_, _| Ok(()))
    }

    fn situ_join(&self) -> ffn::Join {
        ffn::Join::Situ {
            beta: self.hy.situ_beta,
            linear_beta: self.hy.situ_linear_beta,
        }
    }

    /// The output head of row `row` of the last chunk: stream mean, norm, vocabulary matvec.
    fn logits_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        let sc = &self.sc;
        self.io().run(BlockOp::hc_mean(
            row,
            HIDDEN,
            sc.core.resid[0].range(0, (row + 1) * STREAMS * HIDDEN * 4)?,
            sc.head.range(0, HIDDEN * 4)?,
        )?)?;
        let head = head::Head {
            norm: Some(self.io().raw("output_norm.weight")?),
            eps: self.hy.norm_eps,
            weight: head::HeadWeight::Native(self.io().native("output.weight")?),
            softcap: None,
            normed: Rows::both(&sc.core.normed_q8, &sc.core.normed),
            logits: &sc.core.logits,
        };
        head.logits(&self.io(), StepIo::whole(&sc.head)?)?;
        let mut bytes = vec![0u8; self.vocab * 4];
        sc.core.logits.read_at(&self.stream, 0, &mut bytes)?;
        Ok(f32_values(&bytes))
    }
}

impl Model for Raon1Device {
    fn family(&self) -> &'static str {
        "raon1"
    }

    fn layers(&self) -> &[Layer] {
        &self.layers
    }

    fn io(&self, exact: bool) -> StepIo<'_> {
        StepIo {
            family: "raon1",
            ctx: &self.ctx,
            stream: &self.stream,
            weights: &self.weights,
            derived: Some(&self.derived),
            mmq_q8: &self.sc.core.product_q8,
            fixup: &self.sc.core.fixup,
            exact,
        }
    }

    fn exact(&self, _n: usize) -> bool {
        self.exact.get()
    }

    fn core(&self) -> &Core {
        &self.sc.core
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn hidden(&self) -> usize {
        HIDDEN
    }

    fn eps(&self) -> f32 {
        self.hy.norm_eps
    }

    fn delta(&self) -> Result<(delta::Dims, Option<delta::ChannelRows<'_>>), EngError> {
        let sc = &self.sc;
        let rows = delta::ChannelRows {
            low: &sc.fa,
            decay_in: &sc.g1,
            decay: &sc.scan_decay,
            gate_in: &sc.g2,
            beta: &sc.beta,
            delta: &sc.delta,
            strengths: &sc.scan_proj,
        };
        Ok((DELTA, Some(rows)))
    }

    fn latent(
        &self,
        slot: usize,
        _n: usize,
    ) -> Result<(latent::Dims, latent::LatentRows<'_>), EngError> {
        let sc = &self.sc;
        let factors = self.mla.get(slot).ok_or_else(|| {
            EngError::BadRequest(format!("raon1: no latent attention in slot {slot}"))
        })?;
        let layer_cache = self.cells * MLA_ROW * 2;
        let dims = latent::Dims {
            heads: HEADS,
            head_dim: HEAD_DIM,
            key_dim: Q_NOPE,
            rank: RANK,
            rot_dims: ROPE,
            query_rank: Q_LORA,
            scale: 1.0 / (self.spec.key_mla as f32).sqrt(),
            eps: self.hy.latent_eps,
        };
        let rows = latent::LatentRows {
            input: sc.cur_rows(),
            query_low: &sc.qa,
            query_low_q8: &sc.q8_qa,
            query: &sc.qb,
            latent: &sc.kv,
            absorbed: &sc.qeff,
            part: &sc.part,
            mixed: &sc.mix,
            attended: &sc.mla_out,
            gate: &sc.g2,
            gated: Rows::both(&sc.core.read_q8, &sc.mla_gated),
            out: &sc.core.mix_out,
            key_factor: StepIo::whole(&factors.wk_b)?,
            value_factor: StepIo::whole(&factors.wv_b)?,
            cache: self.cache.range(slot * layer_cache, layer_cache)?,
            cells: self.cells,
        };
        Ok((dims, rows))
    }

    fn dense(&self, n: usize) -> Result<(ffn::Join, ffn::Scratch<'_>), EngError> {
        let sc = &self.sc;
        let act = Rows::both(&sc.q8_ffn, &sc.act_ffn);
        Ok((self.situ_join(), sc.ffn_scratch(n, DENSE_INTER, act)?))
    }

    fn routed(
        &self,
        n: usize,
    ) -> Result<((moe::Dims, ffn::Join), &moe::Scratch, ffn::Scratch<'_>), EngError> {
        let sc = &self.sc;
        // The served routes read the rounded rows only; f32 rows beside them would move a
        // chunk onto another product.
        let act = if self.exact.get() {
            Rows::both(&sc.q8_ffn, &sc.act_ffn)
        } else {
            Rows::q8(&sc.q8_ffn)
        };
        let shared = sc.ffn_scratch(n, MOE_INTER, act)?;
        Ok(((MOE, self.situ_join()), &sc.moe, shared))
    }

    fn mixed_site(&self) -> Result<site::Mixed<'_>, EngError> {
        Ok(site::Mixed {
            hidden: HIDDEN,
            eps: self.hy.hc_eps,
            magnitude: self.hy.hc_magnitude,
            input: &self.sc.branch,
            post: &self.sc.post,
            comb: &self.sc.comb,
        })
    }

    /// With a trace on: the normed rows after a read, the kept experts after a routed
    /// feed-forward, and the streams after an apply.
    fn tap(&self, layer: usize, half: Half, stage: WalkStage) -> Result<(), EngError> {
        if self.trace.borrow().is_none() {
            return Ok(());
        }
        let (sc, n) = (&self.sc, self.rows.get());
        let name = |mixer: &str, forward: &str| match half {
            Half::Mixer => format!("{mixer}-{layer}"),
            Half::Forward => format!("{forward}-{layer}"),
        };
        match stage {
            WalkStage::Read => {
                self.snap(&name("attn_norm", "ffn_norm"), &sc.core.normed, n * HIDDEN)
            }
            WalkStage::Branch => {
                if half == Half::Mixer && matches!(self.layers[layer].mixer, Mixer::Latent { .. }) {
                    self.snap("mla_out", &sc.mla_out, n * WIDTH)?;
                    self.snap("mla_gate", &sc.g2, n * WIDTH)?;
                }
                let routed = matches!(self.layers[layer].forward, Forward::Routed(_));
                if half == Half::Forward && routed {
                    self.snap_ids(layer, n)?;
                }
                Ok(())
            }
            WalkStage::Apply => {
                let out = &sc.core.resid[sc.core.cur.get()];
                self.snap(&name("hc_attn_out", "l_out"), out, n * STREAMS * HIDDEN)
            }
        }
    }
}

impl ChunkStep for Raon1Device {
    fn family(&self) -> &'static str {
        "raon1"
    }

    fn chunk_rows(&self) -> usize {
        CHUNK
    }

    fn cells(&self) -> usize {
        self.cells
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn run_chunk(&mut self, tokens: &[u32], pos0: usize) -> Result<(), EngError> {
        self.chunk(tokens, pos0)
    }

    fn head_row(&self, row: usize) -> Result<Vec<f32>, EngError> {
        self.logits_row(row)
    }

    fn drain(&self) -> Result<(), EngError> {
        self.stream.synchronize()
    }
}

/// What the delta-rule layers carry, over a context of `cells` rows.
fn slots(spec: Raon1Spec, cells: usize) -> Slots {
    Slots {
        delta: spec.layers - spec.mla_layers.len(),
        ring: 3 * CONV_TAPS * WIDTH,
        record: HEADS * HEAD_DIM * HEAD_DIM,
        attend: 0,
        plane: 0,
        rows: cells,
    }
}

/// The bytes of the carried state and of the joined convolution taps.
fn state_bytes(spec: Raon1Spec) -> Result<u64, EngError> {
    let plan = Count::default();
    let carried = slots(spec, 0);
    State::new(&plan, carried)?;
    let taps = carried.delta * 3 * WIDTH * SPEC.conv_kernel as usize * 4;
    Ok(plan.total() + taps as u64)
}

/// The scratch of one chunk, declared once: `fixup` is the bytes of the batched product's
/// fix-up words on this device.
fn scratch<A: Alloc>(a: &A, vocab: usize, fixup: usize) -> Result<Scratch<A::Buf>, EngError> {
    let c = CHUNK;
    let f = |count: usize| a.floats(count);
    let q = |width: usize, rows: usize| a.q8(width, rows);
    let part = (c * (HEADS / MLA_GROUP)).max(latent::PART_ROWS) * MLA_GROUP * (RANK + 4);
    let core = Core::new(
        a,
        Widths {
            rows: c,
            hidden: HIDDEN,
            streams: STREAMS,
            proj: 3 * WIDTH,
            conv: 3 * WIDTH,
            read: WIDTH,
            product: DENSE_INTER,
            vocab,
            fixup,
        },
    )?;
    Ok(Scratch {
        core,
        moe: moe::Scratch::new(a, MOE, c)?,
        branch: f(c * HIDDEN)?,
        post: f(c * STREAMS)?,
        comb: f(c * 16)?,
        scan_decay: f(c * WIDTH)?,
        scan_proj: f(c * (WIDTH + HEADS))?,
        g1: f(c * WIDTH)?,
        g2: f(c * WIDTH)?,
        fa: f(c * TIME_RANK)?,
        beta: f(c * HEADS)?,
        delta: f(c * HEADS)?,
        qa: f(c * Q_LORA)?,
        q8_qa: q(Q_LORA, c)?,
        qb: f(c * HEADS * Q_ROW)?,
        kv: f(c * MLA_ROW)?,
        qeff: f(c * HEADS * MLA_ROW)?,
        part: f(part)?,
        mix: f(c * HEADS * RANK)?,
        mla_out: f(c * WIDTH)?,
        ffn_gate: f(c * DENSE_INTER)?,
        ffn_up: f(c * DENSE_INTER)?,
        q8_ffn: q(DENSE_INTER, c)?,
        head: f(HIDDEN)?,
        mla_gated: f(c * WIDTH)?,
        act_ffn: f(c * DENSE_INTER)?,
    })
}
