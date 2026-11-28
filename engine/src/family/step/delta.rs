//! The delta-rule mixer of a visit.
//!
//! Projections write the query, key, value and gate inputs of every row into one strided row. A
//! short causal convolution over its first columns carries its last inputs in a ring, the delta
//! rule updates the layer's record and reads it, and an output projection returns to the hidden
//! width. A layer follows one of two rules. With one decay and one write strength per head, a
//! fused projection writes the inputs and a second one the strengths, one row runs the one-step
//! rule and a chunk the scan that walks its rows in order. With one decay per key channel, the
//! query, key and value have a projection each, low-rank projections write the decay and the
//! strengths, and every chunk runs the scan.

use super::rows::{FEW_ROWS, Map, Out, Rows, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnChannelScanCall, GdnForm, GdnGate, GdnKeyHeads, GdnL2, GdnScanSeqCall,
    ScanShape, ScanWeights,
};
use crate::kernels::cuda::resident::block_ops::{BlockOp, RdaInputs, RdaPacked};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, GDN_EPS};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, NativeWeight};

/// From this many rows a thread of the convolution walks this many rows of its channel.
const CONV_ROWS_PER_THREAD: usize = 16;

const F32: usize = size_of::<f32>();

fn floats(buffer: &DeviceBuffer, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, len * F32)
}

fn f32_rows<'a>(
    io: &StepIo<'_>,
    rows: Option<&'a DeviceBuffer>,
    what: &str,
) -> Result<&'a DeviceBuffer, EngError> {
    rows.ok_or_else(|| {
        EngError::BadRequest(format!(
            "{}: the delta rule needs f32 rows for {what}",
            io.family
        ))
    })
}

/// The weights and the widths of one delta-rule layer.
#[derive(Clone, Copy)]
pub(crate) struct Delta<'a> {
    pub(crate) rule: Rule<'a>,
    pub(crate) out: &'a NativeWeight,
    /// The taps of every convolved column.
    pub(crate) conv: DeviceRange<'a>,
    /// The decay scale of every value head.
    pub(crate) a: DeviceRange<'a>,
    /// The decay bias: one per value head, or one per key channel under [`Rule::Channels`].
    pub(crate) dt_bias: DeviceRange<'a>,
    pub(crate) norm: DeviceRange<'a>,
    pub(crate) key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
    /// Columns of the strided row the convolution covers.
    pub(crate) conv_dim: usize,
}

/// How a layer forms its decay and its write strength.
#[derive(Clone, Copy)]
pub(crate) enum Rule<'a> {
    /// One decay and one write strength per value head. The input projections write one row
    /// side by side, in this order: the query, key and value inputs, the gate input, the write
    /// logit and the decay logit of every head. A file stores them as two matrices (the first
    /// four fused, the two logits fused) or as four.
    Heads {
        inputs: [Option<&'a NativeWeight>; 4],
    },
    /// One decay per key channel.
    Channels(Channels<'a>),
}

/// The weights of the per-channel rule. The F16 matrices read the f32 input rows.
#[derive(Clone, Copy)]
pub(crate) struct Channels<'a> {
    pub(crate) q: &'a NativeWeight,
    pub(crate) k: &'a NativeWeight,
    pub(crate) v: &'a NativeWeight,
    pub(crate) gate: &'a NativeWeight,
    /// The two factors of the decay projection, `rank` wide between them.
    pub(crate) decay_a: DeviceRange<'a>,
    pub(crate) decay_b: DeviceRange<'a>,
    pub(crate) rank: usize,
    /// The write strength logit of every head, and what the value adds to it.
    pub(crate) beta: DeviceRange<'a>,
    pub(crate) delta: DeviceRange<'a>,
    /// The log of the smallest decay.
    pub(crate) lower_bound: f32,
    pub(crate) norm_eps: f32,
}

/// The rows the per-channel rule works in beside [`DeltaRows`].
#[derive(Clone, Copy)]
pub(crate) struct ChannelRows<'a> {
    /// Rows of `rank` floats.
    pub(crate) low: &'a DeviceBuffer,
    /// Rows of one float per channel: the decay logits, then the decay, and the gate inputs.
    pub(crate) decay_in: &'a DeviceBuffer,
    pub(crate) decay: &'a DeviceBuffer,
    pub(crate) gate_in: &'a DeviceBuffer,
    /// Rows of one float per head.
    pub(crate) beta: &'a DeviceBuffer,
    pub(crate) delta: &'a DeviceBuffer,
    /// The `[gate | strength]` rows the scan reads.
    pub(crate) strengths: &'a DeviceBuffer,
}

/// The widths of a delta-rule layer.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub(crate) key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) conv_dim: usize,
}

/// The tensors of one delta-rule layer by their names in the weight map.
pub(crate) struct DeltaNames {
    pub(crate) rule: RuleNames,
    pub(crate) out: String,
    pub(crate) conv: String,
    pub(crate) a: String,
    pub(crate) dt_bias: String,
    pub(crate) norm: String,
}

pub(crate) enum RuleNames {
    /// The input projections in row order, two names or four.
    Heads { inputs: Vec<String> },
    Channels {
        /// The query, key, value and gate projections.
        inputs: [String; 4],
        decay_a: String,
        decay_b: String,
        rank: usize,
        beta: String,
        delta: String,
        lower_bound: f32,
        norm_eps: f32,
    },
}

impl DeltaNames {
    /// The names a GGUF gives the layer of block `stem` under [`RuleNames::Heads`].
    pub(crate) fn fused(stem: &str) -> Self {
        let n = |name: &str| format!("{stem}.{name}");
        Self {
            rule: RuleNames::Heads {
                inputs: vec![n("attn_qkv_gate.weight"), n("ssm_beta_alpha.weight")],
            },
            ..Self::split(stem)
        }
    }

    /// The names of a file that stores the four input projections apart.
    pub(crate) fn split(stem: &str) -> Self {
        let n = |name: &str| format!("{stem}.{name}");
        Self {
            rule: RuleNames::Heads {
                inputs: ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"]
                    .map(|name| n(&format!("{name}.weight")))
                    .to_vec(),
            },
            out: n("ssm_out.weight"),
            conv: n("ssm_conv1d.weight"),
            a: n("ssm_a"),
            dt_bias: n("ssm_dt.bias"),
            norm: n("ssm_norm.weight"),
        }
    }

    /// Every tensor the layer names.
    pub(crate) fn tensors(&self) -> Vec<&str> {
        let mut names: Vec<&str> = match &self.rule {
            RuleNames::Heads { inputs } => inputs.iter().map(String::as_str).collect(),
            RuleNames::Channels {
                inputs,
                decay_a,
                decay_b,
                beta,
                delta,
                ..
            } => inputs
                .iter()
                .chain([decay_a, decay_b, beta, delta])
                .map(String::as_str)
                .collect(),
        };
        names.extend(
            [&self.out, &self.conv, &self.a, &self.dt_bias, &self.norm].map(String::as_str),
        );
        names
    }

    /// The layer over the tensors of `map`.
    pub(crate) fn resolve<'a>(&self, map: &Map<'a>, dims: Dims) -> Result<Delta<'a>, EngError> {
        let rule = match &self.rule {
            RuleNames::Heads { inputs: names } => {
                let mut inputs = [None; 4];
                if names.len() > inputs.len() {
                    return Err(EngError::BadRequest(format!(
                        "{}: a delta layer names {} input projections",
                        map.family,
                        names.len()
                    )));
                }
                for (slot, name) in inputs.iter_mut().zip(names) {
                    *slot = Some(map.native(name)?);
                }
                let rows: usize = inputs.iter().flatten().map(|w| w.n_rows()).sum();
                let want = dims.conv_dim + dims.value_heads * (dims.value_dim + 2);
                if rows != want {
                    return Err(EngError::BadRequest(format!(
                        "{}: the delta input projections write {rows} columns, the rule reads {want}",
                        map.family
                    )));
                }
                Rule::Heads { inputs }
            }
            RuleNames::Channels {
                inputs: [q, k, v, gate],
                decay_a,
                decay_b,
                rank,
                beta,
                delta,
                lower_bound,
                norm_eps,
            } => Rule::Channels(Channels {
                q: map.native(q)?,
                k: map.native(k)?,
                v: map.native(v)?,
                gate: map.native(gate)?,
                decay_a: map.raw(decay_a)?,
                decay_b: map.raw(decay_b)?,
                rank: *rank,
                beta: map.raw(beta)?,
                delta: map.raw(delta)?,
                lower_bound: *lower_bound,
                norm_eps: *norm_eps,
            }),
        };
        Ok(Delta {
            rule,
            out: map.native(&self.out)?,
            conv: map.raw(&self.conv)?,
            a: map.raw(&self.a)?,
            dt_bias: map.raw(&self.dt_bias)?,
            norm: map.raw(&self.norm)?,
            key_heads: dims.key_heads,
            key_dim: dims.key_dim,
            value_heads: dims.value_heads,
            value_dim: dims.value_dim,
            conv_kernel: dims.conv_kernel,
            conv_dim: dims.conv_dim,
        })
    }
}

impl<'a> Delta<'a> {
    /// The quantized projections of the layer, inputs first and the output last.
    #[cfg(test)]
    pub(crate) fn projections(&self) -> impl Iterator<Item = &'a NativeWeight> {
        let inputs = match self.rule {
            Rule::Heads { inputs } => inputs,
            Rule::Channels(c) => [Some(c.q), Some(c.k), Some(c.v), Some(c.gate)],
        };
        inputs.into_iter().flatten().chain([self.out])
    }

    /// Floats of the convolution ring of one visit.
    pub(crate) fn ring_floats(&self) -> usize {
        self.conv_dim * (self.conv_kernel - 1)
    }

    /// Floats of the record of one visit.
    pub(crate) fn record_floats(&self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

/// The rows a delta-rule visit works in.
#[derive(Clone, Copy)]
pub(crate) struct DeltaRows<'a> {
    /// The normed input rows.
    pub(crate) input: Rows<'a>,
    /// The strided rows the projections write.
    pub(crate) proj: &'a DeviceBuffer,
    pub(crate) conv_out: &'a DeviceBuffer,
    /// What the rule reads from the record, and its q8_1 rows.
    pub(crate) read: Rows<'a>,
    /// The mixer output rows, hidden wide.
    pub(crate) out: &'a DeviceBuffer,
    /// The rows of the per-channel rule, for a layer that follows it.
    pub(crate) channel: Option<ChannelRows<'a>>,
}

/// The carried state of one visit: its convolution ring and its record.
pub(crate) struct DeltaState<'a> {
    pub(crate) ring: DeviceRange<'a>,
    pub(crate) record: DeviceRange<'a>,
    /// Rows of the context the scan may reach.
    pub(crate) context: usize,
}

/// One delta-rule visit over `n` rows whose first row sits at `p0`.
pub(crate) fn mix(
    io: &StepIo<'_>,
    layer: &Delta<'_>,
    rows: &DeltaRows<'_>,
    state: DeltaState<'_>,
    n: usize,
    p0: usize,
) -> Result<(), EngError> {
    let (vh, vd, conv_dim) = (layer.value_heads, layer.value_dim, layer.conv_dim);
    let (hidden, read_floats) = (layer.out.n_rows(), vh * vd);
    let read_f32 = f32_rows(io, rows.read.f32, "what it reads")?;
    // The row the convolution reads: `ld` floats apart, the projections at their columns.
    let (ld, parts) = match layer.rule {
        Rule::Heads { inputs } => (inputs.iter().flatten().map(|w| w.n_rows()).sum(), inputs),
        Rule::Channels(c) => (conv_dim, [Some(c.q), Some(c.k), Some(c.v), None]),
    };
    let roles = match layer.rule {
        Rule::Heads { .. } => ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"],
        Rule::Channels(_) => ["attn_qkv", "attn_qkv", "attn_qkv", ""],
    };
    let proj = floats(rows.proj, n * ld)?;
    let mut col_off = 0;
    for (role, weight) in roles
        .into_iter()
        .zip(parts)
        .filter_map(|(r, w)| w.map(|w| (r, w)))
    {
        let out = Out {
            dst: proj,
            ld,
            col_off,
        };
        io.project_role_into(role, weight, rows.input, n, out)?;
        col_off += weight.n_rows();
    }
    let conv_call = ConvCall {
        shape: ConvShape {
            m: n,
            channels: conv_dim,
            ld,
        },
        // One row hands over the convolved columns alone.
        proj: floats(rows.proj, if n == 1 { conv_dim } else { n * ld })?,
        weights: layer.conv.slice(0, conv_dim * layer.conv_kernel * F32)?,
        state: state.ring,
        out: floats(rows.conv_out, n * conv_dim)?,
    };
    if n >= CONV_ROWS_PER_THREAD {
        io.ctx
            .gdn_conv_rows(io.stream, conv_call, CONV_ROWS_PER_THREAD)?;
    } else {
        io.ctx.gdn_conv(io.stream, conv_call)?;
    }
    let (conv, read) = (
        floats(rows.conv_out, n * conv_dim)?,
        floats(read_f32, n * read_floats)?,
    );
    let (a, norm_w) = (layer.a.slice(0, vh * F32)?, layer.norm.slice(0, vd * F32)?);
    let shape = |ld: usize, z_off: usize, beta_off: usize, alpha_off: usize| ScanShape {
        m: n,
        p0: p0 as i64,
        ctx_tokens: state.context,
        n_k_heads: layer.key_heads,
        n_v_heads: vh,
        head_dim: vd,
        conv_ld: conv_dim,
        ld,
        z_off,
        beta_off,
        alpha_off,
    };
    match layer.rule {
        Rule::Heads { .. } => {
            // The two logits follow the convolved columns and the gate input.
            let fused = conv_dim + read_floats;
            let dt_bias = layer.dt_bias.slice(0, vh * F32)?;
            // A few rows run the one-step rule row after row, so each has the bits of a step
            // of its own.
            if n <= FEW_ROWS {
                io.ctx.gdn_delta(
                    io.stream,
                    DeltaCall {
                        shape: DeltaShape {
                            m: n,
                            n_k_heads: layer.key_heads,
                            n_v_heads: vh,
                            conv_ld: conv_dim,
                            ld,
                            z_off: conv_dim,
                            beta_off: fused,
                            alpha_off: fused + vh,
                        },
                        eps: GDN_EPS,
                        conv,
                        proj,
                        weights: DeltaWeights { a, dt_bias, norm_w },
                        state: state.record,
                        out: read,
                    },
                )?;
            } else {
                io.ctx.gdn_scan_seq(
                    io.stream,
                    GdnScanSeqCall {
                        shape: shape(ld, conv_dim, fused, fused + vh),
                        eps: GDN_SCAN_EPS,
                        conv,
                        proj,
                        weights: ScanWeights { a, dt_bias, norm_w },
                        state: state.record,
                        o_out: read,
                        status: None,
                    },
                )?;
            }
        }
        Rule::Channels(c) => {
            let x = f32_rows(io, rows.input.f32, "its low-rank projections")?;
            let Some(r) = rows.channel else {
                return Err(EngError::BadRequest(format!(
                    "{}: the per-channel delta rule has no rows to work in",
                    io.family
                )));
            };
            let (width, in_dim) = (read_floats, c.q.k());
            io.project(c.gate, rows.input, r.gate_in, n)?;
            io.f16_project(c.decay_a, x, r.low, n, c.rank, in_dim)?;
            io.f16_project(c.decay_b, r.low, r.decay_in, n, width, c.rank)?;
            io.f16_project(c.beta, x, r.beta, n, vh, in_dim)?;
            io.f16_project(c.delta, x, r.delta, n, vh, in_dim)?;
            let (decay, strengths) = (
                floats(r.decay, n * width)?,
                floats(r.strengths, n * (width + vh))?,
            );
            io.run(BlockOp::rda_pack(
                n,
                vh,
                vd,
                c.lower_bound,
                RdaInputs {
                    g1: floats(r.decay_in, n * width)?,
                    dt_bias: layer.dt_bias.slice(0, width * F32)?,
                    a,
                    beta: floats(r.beta, n * vh)?,
                    delta: floats(r.delta, n * vh)?,
                    g2: floats(r.gate_in, n * width)?,
                },
                RdaPacked {
                    qkv: conv,
                    decay,
                    proj: strengths,
                },
            )?)?;
            // The q and k L2 norms in the ggml form, a sigmoid output gate over the per-head
            // RMS norm.
            io.ctx.gdn_scan_seq_channel_decay(
                io.stream,
                GdnChannelScanCall {
                    shape: shape(width + vh, 0, width, width),
                    eps: GDN_SCAN_EPS,
                    norm_eps: c.norm_eps,
                    conv,
                    proj: strengths,
                    decay,
                    norm_w,
                    state: state.record,
                    o_out: read,
                    status: None,
                },
                GdnForm {
                    key_heads: GdnKeyHeads::Tiled,
                    l2: GdnL2::Ggml,
                    gate: GdnGate::Sigmoid,
                },
            )?;
        }
    }
    if io.reads_q8(layer.out, rows.read, n) {
        // Each rule keeps the quantizer its rows were first served with: the two round a tie
        // differently, and a model's bytes follow the one it reads.
        match layer.rule {
            Rule::Heads { .. } => io.ctx.quantize_q8_1(
                io.stream,
                QuantizeCall {
                    x: read,
                    y: rows.read.q8.range(0, q8(read_floats, n))?,
                    k: read_floats,
                    ncols: n,
                },
            )?,
            Rule::Channels(_) => {
                io.quantize(n, read_floats, read_f32, None, Rows::q8(rows.read.q8))?;
            }
        }
    }
    io.project_into(layer.out, rows.read, n, Out::packed(rows.out, n, hidden)?)
}
