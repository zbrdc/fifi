//! Host f32 reference forward for the stepping-stone family `Qwen3.6-35B-A3B` (Q36-HR).
//!
//! Activations stay in host `f32` vectors. Reductions accumulate in `f64`. Projections go through
//! the [`Projector`] seam. The block rules follow mlx-lm 0.31.0 (`qwen3_5.py`, `qwen3_next.py`,
//! `gated_delta.py`), which the checkpoint was converted for. They differ from the Flash-Next host
//! blocks in these points: norm weights apply as stored (no `1 +`), the DeltaNet output gate is
//! `silu(z)`, the DeltaNet query and key norm uses `rms_norm(x, None, 1e-6)` (so the epsilon acts as
//! `D * 1e-6` against the sum of squares), attention is dense with a sigmoid output gate, and the
//! residual stream has one branch.
//!
//! This code is a reference for later packets. It favors plain loops over speed.

use crate::EngError;
use crate::family::flash_next::math::{expect_len, rope_half, sigmoid, silu, softplus};
use crate::family::project::Projector;
use crate::family::route::route_experts;
use crate::family::spec::{AttentionKind, FamilySpec, Router};

/// Epsilon of the DeltaNet query and key norm: the literal `1e-6` of `rms_norm(q, None, 1e-6)`
/// (`qwen3_5.py` line 174), not the configuration epsilon.
const QK_RMS_EPS: f32 = 1e-6;

/// DeltaNet dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaDims {
    pub(crate) key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
}

impl DeltaDims {
    /// Channels of the fused query, key, and value projection.
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }
}

/// Dense attention dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AttnDims {
    pub(crate) heads_q: usize,
    pub(crate) heads_kv: usize,
    pub(crate) head_dim: usize,
    /// Leading dimensions of each head that RoPE rotates.
    pub(crate) rotary_dim: usize,
    pub(crate) theta: f32,
}

/// Mixture-of-experts dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MoeDims {
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    pub(crate) inter: usize,
    pub(crate) shared_inter: usize,
    pub(crate) renorm: bool,
}

/// Checked dimensions of the whole forward pass.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostShape {
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    /// One entry per layer: `Full` or `DeltaNet`.
    pub(crate) kinds: Vec<AttentionKind>,
    /// Epsilon of every learned-weight RMS norm.
    pub(crate) eps: f32,
    pub(crate) delta: DeltaDims,
    pub(crate) attn: AttnDims,
    /// Dense intermediate width, or `None` for MoE layers.
    pub(crate) dense_inter: Option<usize>,
    /// All four widths/counts are zero sentinels for dense layers. Dense loading never uses them.
    pub(crate) moe: MoeDims,
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("host shape: {message}"))
}

fn to_usize(value: u32) -> usize {
    // `usize` holds a `u32` on every supported target.
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl HostShape {
    /// Builds the shape from a family specification and the RoPE dimension the configuration gives.
    pub(crate) fn from_spec(spec: &FamilySpec, rotary_dim: u32) -> Result<Self, EngError> {
        let linear = spec
            .linear
            .ok_or_else(|| invalid("the family has no linear attention dimensions"))?;
        let Router::SoftmaxTopK { renorm } = spec.router else {
            return Err(invalid("the family router is not softmax top-k"));
        };
        let shape = Self {
            hidden: to_usize(spec.hidden),
            vocab: to_usize(spec.vocab),
            kinds: spec.layers_attention.to_vec(),
            eps: spec.rms_norm_eps,
            delta: DeltaDims {
                key_heads: to_usize(linear.key_heads),
                key_dim: to_usize(linear.key_head_dim),
                value_heads: to_usize(linear.value_heads),
                value_dim: to_usize(linear.value_head_dim),
                conv_kernel: to_usize(linear.conv_kernel),
            },
            attn: AttnDims {
                heads_q: to_usize(spec.heads_q),
                heads_kv: to_usize(spec.heads_kv),
                head_dim: to_usize(spec.head_dim),
                rotary_dim: to_usize(rotary_dim),
                theta: spec.rope_theta,
            },
            dense_inter: spec.feed_forward.map(|dense| to_usize(dense.intermediate)),
            moe: MoeDims {
                experts: to_usize(spec.experts),
                top_k: to_usize(spec.top_k),
                inter: to_usize(spec.moe_inter),
                shared_inter: to_usize(spec.shared_inter),
                renorm,
            },
        };
        shape.check()?;
        Ok(shape)
    }

    /// Checks active dimensions and head ratios. Dense layers require zero MoE sentinels.
    pub(crate) fn check(&self) -> Result<(), EngError> {
        let d = &self.delta;
        let a = &self.attn;
        let m = &self.moe;
        let positive = [
            self.hidden,
            self.vocab,
            d.key_heads,
            d.key_dim,
            d.value_heads,
            d.value_dim,
            a.heads_q,
            a.heads_kv,
            a.head_dim,
        ];
        if self.kinds.is_empty() || positive.contains(&0) {
            return Err(invalid("a dimension is zero"));
        }
        let moe_dims = [m.experts, m.top_k, m.inter, m.shared_inter];
        match self.dense_inter {
            Some(0) => return Err(invalid("dense intermediate dimension is zero")),
            Some(_) if moe_dims != [0; 4] => {
                return Err(invalid("dense layers require zero MoE dimensions"));
            }
            None if moe_dims.contains(&0) => return Err(invalid("a dimension is zero")),
            _ => {}
        }
        if self.kinds.contains(&AttentionKind::Qsa) {
            return Err(invalid("the stepping stone has no QSA layer"));
        }
        if d.conv_kernel < 2 || !d.value_heads.is_multiple_of(d.key_heads) {
            return Err(invalid("DeltaNet kernel or head ratio is invalid"));
        }
        if !a.heads_q.is_multiple_of(a.heads_kv) {
            return Err(invalid("query heads must divide by key-value heads"));
        }
        if a.rotary_dim == 0 || !a.rotary_dim.is_multiple_of(2) || a.rotary_dim > a.head_dim {
            return Err(invalid("rotary dimension must be even and within the head"));
        }
        if !a.theta.is_finite() || a.theta <= 0.0 || !self.eps.is_finite() || self.eps < 0.0 {
            return Err(invalid("theta or epsilon is invalid"));
        }
        if m.top_k > m.experts {
            return Err(invalid("top-k exceeds the expert count"));
        }
        Ok(())
    }
}

/// Weights of one DeltaNet layer. `W` is the projector's weight handle.
pub(crate) struct DeltaW<'a, W> {
    /// Fused projection `[conv_dim, hidden]`.
    pub(crate) qkv: &'a W,
    /// Output gate projection `[value_heads * value_dim, hidden]`.
    pub(crate) z: &'a W,
    /// Beta projection `[value_heads, hidden]`.
    pub(crate) b: &'a W,
    /// Alpha projection `[value_heads, hidden]`.
    pub(crate) a: &'a W,
    /// Output projection `[hidden, value_heads * value_dim]`.
    pub(crate) out: &'a W,
    /// Convolution taps, `conv_dim * conv_kernel` values, channel `c` at `c * conv_kernel`.
    pub(crate) conv: &'a [f32],
    /// `A_log`, one value per value head.
    pub(crate) a_log: &'a [f32],
    /// `dt_bias`, one value per value head.
    pub(crate) dt_bias: &'a [f32],
    /// Output norm weight, `value_dim` values shared by all value heads.
    pub(crate) norm: &'a [f32],
}

/// Weights of one dense attention layer.
pub(crate) struct AttnW<'a, W> {
    /// Query and gate projection `[heads_q * head_dim * 2, hidden]`; each head holds its query rows, then its gate rows.
    pub(crate) q: &'a W,
    /// Key projection `[heads_kv * head_dim, hidden]`.
    pub(crate) k: &'a W,
    /// Value projection `[heads_kv * head_dim, hidden]`.
    pub(crate) v: &'a W,
    /// Output projection `[hidden, heads_q * head_dim]`.
    pub(crate) o: &'a W,
    /// Query norm weight, `head_dim` values.
    pub(crate) q_norm: &'a [f32],
    /// Key norm weight, `head_dim` values.
    pub(crate) k_norm: &'a [f32],
}

/// Weights of one MoE layer.
pub(crate) struct MoeW<'a, W> {
    /// Router `[experts, hidden]`.
    pub(crate) router: &'a W,
    /// Expert gate stack `[experts, inter, hidden]`.
    pub(crate) gate: &'a W,
    /// Expert up stack `[experts, inter, hidden]`.
    pub(crate) up: &'a W,
    /// Expert down stack `[experts, hidden, inter]`.
    pub(crate) down: &'a W,
    /// Shared expert gate `[shared_inter, hidden]`.
    pub(crate) shared_gate: &'a W,
    /// Shared expert up `[shared_inter, hidden]`.
    pub(crate) shared_up: &'a W,
    /// Shared expert down `[hidden, shared_inter]`.
    pub(crate) shared_down: &'a W,
    /// Shared expert scalar gate `[1, hidden]`.
    pub(crate) shared_scalar: &'a W,
}

/// The token mixer of one layer.
pub(crate) enum MixerW<'a, W> {
    /// A `linear_attention` layer.
    Delta(DeltaW<'a, W>),
    /// A `full_attention` layer.
    Attn(AttnW<'a, W>),
}

/// Packed dense feed-forward projections: `[inter, hidden]`, `[inter, hidden]`, `[hidden, inter]`.
pub(crate) struct DenseW<'a, W> {
    pub(crate) gate: &'a W,
    pub(crate) up: &'a W,
    pub(crate) down: &'a W,
}

pub(crate) enum FeedForwardW<'a, W> {
    Dense(DenseW<'a, W>),
    Moe(MoeW<'a, W>),
}

/// Weights of one decoder layer.
pub(crate) struct LayerW<'a, W> {
    /// Norm before the mixer, `hidden` values.
    pub(crate) input_norm: &'a [f32],
    /// Norm before the feed-forward block, `hidden` values.
    pub(crate) post_norm: &'a [f32],
    pub(crate) mixer: MixerW<'a, W>,
    pub(crate) feed_forward: FeedForwardW<'a, W>,
}

impl<'a, W> LayerW<'a, W> {
    pub(crate) fn as_moe(&self) -> Result<&MoeW<'a, W>, EngError> {
        match &self.feed_forward {
            FeedForwardW::Moe(moe) => Ok(moe),
            FeedForwardW::Dense(_) => Err(EngError::BadRequest(
                "dense execution is not implemented yet".into(),
            )),
        }
    }
}

/// Weights of the whole model.
pub(crate) struct ModelW<'a, W> {
    /// Embedding table `[vocab, hidden]`.
    pub(crate) embed: &'a W,
    pub(crate) layers: Vec<LayerW<'a, W>>,
    /// Final norm weight, `hidden` values.
    pub(crate) final_norm: &'a [f32],
    /// Output head `[vocab, hidden]`.
    pub(crate) lm_head: &'a W,
}

/// Recurrent state and convolution ring of one DeltaNet layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeltaState {
    /// `[value_heads, value_dim, key_dim]` row-major, the layout of mlx-lm.
    rec: Vec<f32>,
    /// `[conv_dim, conv_kernel - 1]` row-major, oldest tap first.
    ring: Vec<f32>,
}

impl DeltaState {
    /// Returns a zero state.
    pub(crate) fn new(d: &DeltaDims) -> Self {
        Self {
            rec: vec![0.0; d.value_heads * d.value_dim * d.key_dim],
            ring: vec![0.0; d.conv_dim() * (d.conv_kernel - 1)],
        }
    }

    /// Recurrent state values (test observer).
    #[cfg(test)]
    pub(crate) fn rec(&self) -> &[f32] {
        &self.rec
    }
}

/// Keys and values of one dense attention layer, one entry per position.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct KvCache {
    /// `[positions, heads_kv, head_dim]` row-major, after the key norm and RoPE.
    keys: Vec<f32>,
    /// `[positions, heads_kv, head_dim]` row-major.
    values: Vec<f32>,
    positions: usize,
}

/// State of one layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LayerState {
    Delta(DeltaState),
    Attn(KvCache),
}

/// State of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostState {
    layers: Vec<LayerState>,
    pos: u32,
    failed: bool,
}

impl HostState {
    /// Returns a fresh state whose layer kinds follow `shape`.
    pub(crate) fn new(shape: &HostShape) -> Self {
        let layers = shape
            .kinds
            .iter()
            .map(|kind| match kind {
                AttentionKind::DeltaNet => LayerState::Delta(DeltaState::new(&shape.delta)),
                _ => LayerState::Attn(KvCache::default()),
            })
            .collect();
        Self {
            layers,
            pos: 0,
            failed: false,
        }
    }

    /// Position of the next token.
    pub(crate) fn pos(&self) -> u32 {
        self.pos
    }

    /// Layer states in model order (test observer).
    #[cfg(test)]
    pub(crate) fn layers(&self) -> &[LayerState] {
        &self.layers
    }
}

/// Dot product of two equal-length slices, accumulated in `f64`.
fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// `x / sqrt(mean(x^2) + eps)`, the operation of `mx.fast.rms_norm(x, None, eps)`.
pub(crate) fn rms_unit(x: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64;
    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
    x.iter().map(|v| (f64::from(*v) * scale) as f32).collect()
}

/// `x / sqrt(mean(x^2) + eps) * weight`, with the weight applied as stored.
pub(crate) fn rms_plain(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, EngError> {
    expect_len(weight, x.len(), "rms norm weight")?;
    Ok(rms_unit(x, eps)
        .into_iter()
        .zip(weight)
        .map(|(v, w)| v * w)
        .collect())
}

/// Softmax of `x` accumulated in `f64`.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = x.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / sum) as f32).collect()
}

/// Normalizes the query and key of one head: `q / D` and `k / sqrt(D)` times `rms_norm(., None, 1e-6)`.
pub(crate) fn qk_norm(q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let inv_scale = (q.len() as f32).sqrt().recip();
    let q = rms_unit(q, QK_RMS_EPS)
        .into_iter()
        .map(|v| v * inv_scale * inv_scale)
        .collect();
    let k = rms_unit(k, QK_RMS_EPS)
        .into_iter()
        .map(|v| v * inv_scale)
        .collect();
    (q, k)
}

/// One gated delta-rule step of one value head. `rec` holds `value_dim` rows of `key_dim` values.
/// For each row `j`: `S[j] <- decay * S[j]`, `delta = (v[j] - S[j] . k) * beta`, `S[j] <- S[j] + delta * k`,
/// and the output is `S[j] . q`.
fn delta_rule(
    rec: &mut [f32],
    (q, k, v): (&[f32], &[f32], &[f32]),
    (decay, beta): (f32, f32),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    for (row, value) in rec.chunks_mut(k.len()).zip(v) {
        for s in row.iter_mut() {
            *s *= decay;
        }
        let delta = (value - dot64(row, k) as f32) * beta;
        for (s, key) in row.iter_mut().zip(k) {
            *s += key * delta;
        }
        out.push(dot64(row, q) as f32);
    }
    out
}

/// Runs one DeltaNet decode step and returns the layer output of length `hidden`.
///
/// Rule (mlx-lm `GatedDeltaNet.__call__` with `gated_delta_ops`): project, causal depthwise
/// convolution over the ring and the new row, SiLU, split into query, key, and value, normalize
/// query and key, then per value head `S <- g S`, `delta = (v - S k) * beta`, `S <- S + delta (x) k`,
/// `y = S q`; the head output is `silu(z) * rms_norm(y, norm)`. Key head `h / (value_heads / key_heads)`
/// serves value head `h`.
pub(crate) fn delta_step<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    weights: &DeltaW<'_, P::Weight>,
    state: &mut DeltaState,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let d = &shape.delta;
    let (kh, kd, vh, vd, ck) = (
        d.key_heads,
        d.key_dim,
        d.value_heads,
        d.value_dim,
        d.conv_kernel,
    );
    let conv_dim = d.conv_dim();
    let (kw, vw, taps) = (kh * kd, vh * vd, ck - 1);
    expect_len(x, shape.hidden, "delta input")?;
    expect_len(weights.conv, conv_dim * ck, "delta conv")?;
    expect_len(weights.a_log, vh, "delta A_log")?;
    expect_len(weights.dt_bias, vh, "delta dt_bias")?;
    expect_len(weights.norm, vd, "delta norm")?;
    expect_len(&state.rec, vh * vd * kd, "delta state")?;
    expect_len(&state.ring, conv_dim * taps, "delta ring")?;

    let u = projector.linear(weights.qkv, x)?;
    expect_len(&u, conv_dim, "delta qkv")?;
    let z = projector.linear(weights.z, x)?;
    expect_len(&z, vw, "delta z")?;
    let b = projector.linear(weights.b, x)?;
    expect_len(&b, vh, "delta b")?;
    let a = projector.linear(weights.a, x)?;
    expect_len(&a, vh, "delta a")?;

    let mut y = Vec::with_capacity(conv_dim);
    let mut ring = Vec::with_capacity(state.ring.len());
    for (c, &newest) in u.iter().enumerate() {
        let mut acc = weights.conv[c * ck + taps] * newest;
        for t in 0..taps {
            acc += weights.conv[c * ck + t] * state.ring[c * taps + t];
        }
        y.push(silu(acc));
        ring.extend_from_slice(&state.ring[c * taps + 1..(c + 1) * taps]);
        ring.push(newest);
    }

    let mut q = Vec::with_capacity(kw);
    let mut k = Vec::with_capacity(kw);
    for g in 0..kh {
        let (qg, kg) = qk_norm(&y[g * kd..(g + 1) * kd], &y[kw + g * kd..kw + (g + 1) * kd]);
        q.extend(qg);
        k.extend(kg);
    }

    let rep = vh / kh;
    let mut rec = state.rec.clone();
    let mut o = Vec::with_capacity(vw);
    for h in 0..vh {
        let g = h / rep;
        let (qh, kk) = (&q[g * kd..(g + 1) * kd], &k[g * kd..(g + 1) * kd]);
        let decay = (-weights.a_log[h].exp() * softplus(a[h] + weights.dt_bias[h])).exp();
        let beta = sigmoid(b[h]);
        let core = delta_rule(
            &mut rec[h * vd * kd..(h + 1) * vd * kd],
            (qh, kk, &y[2 * kw + h * vd..2 * kw + (h + 1) * vd]),
            (decay, beta),
        );
        let normed = rms_plain(&core, weights.norm, shape.eps)?;
        o.extend((0..vd).map(|j| silu(z[h * vd + j]) * normed[j]));
    }
    let result = projector.linear(weights.out, &o)?;
    expect_len(&result, shape.hidden, "delta output")?;
    state.ring = ring;
    state.rec = rec;
    Ok(result)
}

/// Runs one dense attention decode step at position `pos` and returns the layer output.
///
/// Rule (mlx-lm `Qwen3NextAttention`): the query projection holds, per head, the query rows and then
/// the gate rows. Query and key take a weighted RMS norm per head, then RoPE on the first
/// `rotary_dim` dimensions (half-split pairs). Attention is causal with scale `head_dim^-0.5`, and
/// query head `h` reads key-value head `h / (heads_q / heads_kv)`. The head output is multiplied by
/// `sigmoid(gate)` before the output projection.
pub(crate) fn attn_step<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    weights: &AttnW<'_, P::Weight>,
    cache: &mut KvCache,
    pos: u32,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let a = &shape.attn;
    let (hq, hkv, hd) = (a.heads_q, a.heads_kv, a.head_dim);
    expect_len(x, shape.hidden, "attention input")?;
    expect_len(weights.q_norm, hd, "attention q norm")?;
    expect_len(weights.k_norm, hd, "attention k norm")?;
    if cache.positions != pos as usize {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, step is at {pos}",
            cache.positions
        )));
    }
    let qg = projector.linear(weights.q, x)?;
    expect_len(&qg, hq * hd * 2, "attention q")?;
    let k = projector.linear(weights.k, x)?;
    expect_len(&k, hkv * hd, "attention k")?;
    let v = projector.linear(weights.v, x)?;
    expect_len(&v, hkv * hd, "attention v")?;

    let mut queries = Vec::with_capacity(hq * hd);
    let mut gates = Vec::with_capacity(hq * hd);
    for h in 0..hq {
        let base = h * 2 * hd;
        let mut query = rms_plain(&qg[base..base + hd], weights.q_norm, shape.eps)?;
        rope_half(&mut query, a.rotary_dim, pos, a.theta)?;
        queries.extend(query);
        gates.extend_from_slice(&qg[base + hd..base + 2 * hd]);
    }
    let mut keys = Vec::with_capacity(hkv * hd);
    for g in 0..hkv {
        let mut key = rms_plain(&k[g * hd..(g + 1) * hd], weights.k_norm, shape.eps)?;
        rope_half(&mut key, a.rotary_dim, pos, a.theta)?;
        keys.extend(key);
    }
    cache.keys.extend(keys);
    cache.values.extend_from_slice(&v);
    cache.positions += 1;

    let scale = (hd as f64).sqrt().recip();
    let rep = hq / hkv;
    let at = |t: usize, g: usize| (t * hkv + g) * hd..(t * hkv + g + 1) * hd;
    let mut o = Vec::with_capacity(hq * hd);
    for h in 0..hq {
        let g = h / rep;
        let query = &queries[h * hd..(h + 1) * hd];
        let scores: Vec<f32> = (0..cache.positions)
            .map(|t| (dot64(query, &cache.keys[at(t, g)]) * scale) as f32)
            .collect();
        let weights_t = softmax(&scores);
        for d in 0..hd {
            let sum: f64 = (0..cache.positions)
                .map(|t| f64::from(weights_t[t]) * f64::from(cache.values[at(t, g)][d]))
                .sum();
            o.push(sum as f32);
        }
    }
    for (value, gate) in o.iter_mut().zip(&gates) {
        *value *= sigmoid(*gate);
    }
    let result = projector.linear(weights.o, &o)?;
    expect_len(&result, shape.hidden, "attention output")?;
    Ok(result)
}

/// Runs one MoE step and returns the layer output of length `hidden`.
///
/// Rule (mlx-lm `Qwen3NextSparseMoeBlock`): softmax over the router logits, the top `top_k`
/// experts (the lowest index wins a tie), gates renormalized over the selection, each expert
/// `down(silu(gate x) * up x)`, plus the shared expert times `sigmoid(shared_scalar x)`.
pub(crate) fn moe_step<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    weights: &MoeW<'_, P::Weight>,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let m = &shape.moe;
    expect_len(x, shape.hidden, "moe input")?;
    let logits = projector.linear(weights.router, x)?;
    expect_len(&logits, m.experts, "moe router")?;
    let probs = softmax(&logits);
    let top_k =
        u32::try_from(m.top_k).map_err(|_| EngError::Checkpoint("moe top-k exceeds u32".into()))?;
    let routed = route_experts(&probs, &Router::SoftmaxTopK { renorm: m.renorm }, top_k)?;
    let mut out = vec![0.0f32; shape.hidden];
    for (&expert, &gate) in routed.indices.iter().zip(&routed.weights) {
        let g = projector.expert_linear(weights.gate, expert, x)?;
        expect_len(&g, m.inter, "moe gate output")?;
        let u = projector.expert_linear(weights.up, expert, x)?;
        expect_len(&u, m.inter, "moe up output")?;
        let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        let down = projector.expert_linear(weights.down, expert, &h)?;
        expect_len(&down, shape.hidden, "moe down output")?;
        for (acc, v) in out.iter_mut().zip(&down) {
            *acc += gate * v;
        }
    }
    let sg = projector.linear(weights.shared_gate, x)?;
    expect_len(&sg, m.shared_inter, "moe shared gate output")?;
    let su = projector.linear(weights.shared_up, x)?;
    expect_len(&su, m.shared_inter, "moe shared up output")?;
    let sh: Vec<f32> = sg.iter().zip(&su).map(|(a, b)| silu(*a) * b).collect();
    let shared = projector.linear(weights.shared_down, &sh)?;
    expect_len(&shared, shape.hidden, "moe shared down output")?;
    let scalar = projector.linear(weights.shared_scalar, x)?;
    expect_len(&scalar, 1, "moe shared scalar")?;
    let factor = sigmoid(scalar[0]);
    for (acc, v) in out.iter_mut().zip(&shared) {
        *acc += factor * v;
    }
    Ok(out)
}

/// Result of one decode step.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostStep {
    /// Next-token logits, `vocab` values.
    pub(crate) logits: Vec<f32>,
    /// The residual stream after each layer, when the caller asked to record it; empty otherwise.
    pub(crate) layer_outputs: Vec<Vec<f32>>,
}

/// Runs one layer: `h = x + mixer(norm(x))`, then `h + moe(norm(h))`.
fn layer_step<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    layer: &LayerW<'_, P::Weight>,
    state: &mut LayerState,
    pos: u32,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let normed = rms_plain(x, layer.input_norm, shape.eps)?;
    let mixed = match (&layer.mixer, state) {
        (MixerW::Delta(w), LayerState::Delta(s)) => delta_step(projector, shape, w, s, &normed)?,
        (MixerW::Attn(w), LayerState::Attn(c)) => attn_step(projector, shape, w, c, pos, &normed)?,
        _ => {
            return Err(EngError::BadRequest(
                "layer weights and state differ".into(),
            ));
        }
    };
    let h: Vec<f32> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();
    let normed = rms_plain(&h, layer.post_norm, shape.eps)?;
    let routed = moe_step(projector, shape, layer.as_moe()?, &normed)?;
    Ok(h.iter().zip(&routed).map(|(a, b)| a + b).collect())
}

/// Runs one decode step for `token` at the position `state` holds and advances the state.
///
/// After an error the state is invalid: every later step returns `BadRequest`.
pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    weights: &ModelW<'_, P::Weight>,
    state: &mut HostState,
    token: u32,
    record: bool,
) -> Result<HostStep, EngError> {
    if state.failed {
        return Err(EngError::BadRequest(
            "the host state is invalid after an error".into(),
        ));
    }
    let result = step_inner(projector, shape, weights, state, token, record);
    if result.is_err() {
        state.failed = true;
    }
    result
}

fn step_inner<P: Projector>(
    projector: &mut P,
    shape: &HostShape,
    weights: &ModelW<'_, P::Weight>,
    state: &mut HostState,
    token: u32,
    record: bool,
) -> Result<HostStep, EngError> {
    if to_usize(token) >= shape.vocab {
        return Err(EngError::BadRequest(format!(
            "token {token} is outside the vocabulary of {}",
            shape.vocab
        )));
    }
    if weights.layers.len() != shape.kinds.len() || state.layers.len() != shape.kinds.len() {
        return Err(EngError::BadRequest(
            "layer counts of shape, weights, and state differ".into(),
        ));
    }
    let mut x = projector.rows(weights.embed, &[token])?;
    expect_len(&x, shape.hidden, "embedding row")?;
    let mut layer_outputs = Vec::new();
    for (layer, layer_state) in weights.layers.iter().zip(state.layers.iter_mut()) {
        x = layer_step(projector, shape, layer, layer_state, state.pos, &x)?;
        if record {
            layer_outputs.push(x.clone());
        }
    }
    let normed = rms_plain(&x, weights.final_norm, shape.eps)?;
    let logits = projector.linear(weights.lm_head, &normed)?;
    expect_len(&logits, shape.vocab, "logits")?;
    state.pos = state
        .pos
        .checked_add(1)
        .ok_or_else(|| EngError::BadRequest("position overflows u32".into()))?;
    Ok(HostStep {
        logits,
        layer_outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};
    use crate::family::qwen36::fixtures::val;
    use crate::family::spec::QWEN3_6_35B_A3B;

    // The fixtures below follow the Q36 oracle script (`q36_oracle.py`). Weights come from `val`, an
    // integer hash with dyadic results, so the Rust tests and the f64 oracle hold identical values.

    fn vals(tag: u64, n: usize) -> Vec<f32> {
        (0..n as u64).map(|i| val(tag, i)).collect()
    }

    fn matrix(tag: u64, rows: usize, cols: usize) -> DenseWeight {
        DenseWeight::matrix(rows, cols, vals(tag, rows * cols))
    }

    fn stack(tag: u64, depth: usize, rows: usize, cols: usize) -> DenseWeight {
        DenseWeight::stack(depth, rows, cols, vals(tag, depth * rows * cols))
    }

    /// `1 + val / 4`, the norm weights of the oracle.
    fn norm(tag: u64, n: usize) -> Vec<f32> {
        vals(tag, n).iter().map(|v| 1.0 + v / 4.0).collect()
    }

    fn halves(tag: u64, n: usize) -> Vec<f32> {
        vals(tag, n).iter().map(|v| v / 2.0).collect()
    }

    fn close(actual: f32, expected: f64) -> bool {
        (f64::from(actual) - expected).abs() <= 1e-5 + 1e-5 * expected.abs()
    }

    fn assert_close(actual: &[f32], expected: &[f64], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(close(*a, *e), "{what}[{i}]: {a} vs {e}");
        }
    }

    const HIDDEN: usize = 4;

    fn delta_dims() -> DeltaDims {
        DeltaDims {
            key_heads: 2,
            key_dim: 2,
            value_heads: 4,
            value_dim: 2,
            conv_kernel: 4,
        }
    }

    fn attn_dims() -> AttnDims {
        AttnDims {
            heads_q: 4,
            heads_kv: 2,
            head_dim: 4,
            rotary_dim: 2,
            theta: 100.0,
        }
    }

    fn moe_dims() -> MoeDims {
        MoeDims {
            experts: 4,
            top_k: 2,
            inter: 3,
            shared_inter: 2,
            renorm: true,
        }
    }

    fn shape(kinds: Vec<AttentionKind>) -> HostShape {
        HostShape {
            hidden: HIDDEN,
            vocab: 6,
            kinds,
            eps: 1e-6,
            delta: delta_dims(),
            attn: attn_dims(),
            dense_inter: None,
            moe: moe_dims(),
        }
    }

    /// Owned weights of one DeltaNet layer; the oracle `Delta(tag, ...)` uses the same tags.
    struct DeltaFx {
        qkv: DenseWeight,
        z: DenseWeight,
        b: DenseWeight,
        a: DenseWeight,
        out: DenseWeight,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
    }

    impl DeltaFx {
        fn new(tag: u64) -> Self {
            Self {
                qkv: matrix(tag + 1, 16, HIDDEN),
                z: matrix(tag + 2, 8, HIDDEN),
                b: matrix(tag + 3, 4, HIDDEN),
                a: matrix(tag + 4, 4, HIDDEN),
                out: matrix(tag + 5, HIDDEN, 8),
                conv: vals(tag + 6, 16 * 4),
                a_log: halves(tag + 7, 4),
                dt_bias: halves(tag + 8, 4),
                norm: norm(tag + 9, 2),
            }
        }

        fn weights(&self) -> DeltaW<'_, DenseWeight> {
            DeltaW {
                qkv: &self.qkv,
                z: &self.z,
                b: &self.b,
                a: &self.a,
                out: &self.out,
                conv: &self.conv,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                norm: &self.norm,
            }
        }
    }

    struct AttnFx {
        q: DenseWeight,
        k: DenseWeight,
        v: DenseWeight,
        o: DenseWeight,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }

    impl AttnFx {
        fn new(tag: u64) -> Self {
            Self {
                q: matrix(tag + 1, 32, HIDDEN),
                k: matrix(tag + 2, 8, HIDDEN),
                v: matrix(tag + 3, 8, HIDDEN),
                o: matrix(tag + 4, HIDDEN, 16),
                q_norm: norm(tag + 5, 4),
                k_norm: norm(tag + 6, 4),
            }
        }

        fn weights(&self) -> AttnW<'_, DenseWeight> {
            AttnW {
                q: &self.q,
                k: &self.k,
                v: &self.v,
                o: &self.o,
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
            }
        }
    }

    struct MoeFx {
        router: DenseWeight,
        gate: DenseWeight,
        up: DenseWeight,
        down: DenseWeight,
        shared_gate: DenseWeight,
        shared_up: DenseWeight,
        shared_down: DenseWeight,
        shared_scalar: DenseWeight,
    }

    impl MoeFx {
        fn new(tag: u64) -> Self {
            Self {
                router: matrix(tag + 1, 4, HIDDEN),
                gate: stack(tag + 2, 4, 3, HIDDEN),
                up: stack(tag + 3, 4, 3, HIDDEN),
                down: stack(tag + 4, 4, HIDDEN, 3),
                shared_gate: matrix(tag + 5, 2, HIDDEN),
                shared_up: matrix(tag + 6, 2, HIDDEN),
                shared_down: matrix(tag + 7, HIDDEN, 2),
                shared_scalar: matrix(tag + 8, 1, HIDDEN),
            }
        }

        fn weights(&self) -> MoeW<'_, DenseWeight> {
            MoeW {
                router: &self.router,
                gate: &self.gate,
                up: &self.up,
                down: &self.down,
                shared_gate: &self.shared_gate,
                shared_up: &self.shared_up,
                shared_down: &self.shared_down,
                shared_scalar: &self.shared_scalar,
            }
        }
    }

    /// Oracle: DeltaNet output at steps 0, 1, 2 (`Delta` of `q36_oracle.py`).
    const DELTA_OUT: [[f64; 4]; 3] = [
        [1.59505406, 1.187938048, -2.742731905, -0.345894528],
        [-0.407854233, 0.029660382, -0.42687579, 0.100670465],
        [2.176664118, -1.576373533, -1.596136109, 1.68859448],
    ];

    /// Oracle: recurrent state `[4 heads][2 value][2 key]` after step 2.
    const DELTA_STATE: [f64; 16] = [
        -0.026955212,
        0.027617244,
        0.125316123,
        -0.074006013,
        0.026486412,
        0.004441972,
        0.12575979,
        -0.035785094,
        -0.08839821,
        0.124378641,
        0.329203412,
        0.380372016,
        -0.260245525,
        -0.014653108,
        0.005780345,
        0.011036557,
    ];

    /// Oracle: attention output at positions 0, 1, 2.
    const ATTN_OUT: [[f64; 4]; 3] = [
        [0.205813718, 0.569388379, 0.555566258, -0.4295172],
        [0.000286779, 0.903932202, -0.811443402, -0.576138059],
        [0.529288096, 0.591187571, 0.539563978, 0.039145148],
    ];

    /// Oracle: attention output at positions 0, 1, 2 with rotary_dim 4 (all of head_dim), so theta matters.
    const ATTN_OUT_ROT4: [[f64; 4]; 3] = [
        [0.205813718, 0.569388379, 0.555566258, -0.4295172],
        [-0.12270037, 1.006483214, -1.216035889, -0.728584796],
        [0.887909449, -0.018695252, 1.084390244, 0.336585386],
    ];

    /// Oracle: MoE output for the inputs 120, 121, 122; the experts chosen are [[2, 3], [1, 0], [3, 0]].
    const MOE_OUT: [[f64; 4]; 3] = [
        [-0.062577223, 0.083327072, -0.094178786, 0.191901215],
        [0.407411755, 0.781365809, -0.081587005, 0.390379222],
        [-0.598096521, -0.218829429, -0.282262561, -0.030608096],
    ];

    /// Oracle: logits for tokens 1, 4, 2.
    const MODEL_LOGITS: [[f64; 6]; 3] = [
        [
            1.277054376,
            -0.245087558,
            0.848396859,
            -0.385288982,
            -1.494347517,
            0.821287024,
        ],
        [
            1.409500528,
            1.485335916,
            -0.749353929,
            -0.381608256,
            -1.457096093,
            1.413156168,
        ],
        [
            -1.145210117,
            0.427164675,
            -0.244757922,
            1.208101975,
            0.47105159,
            -0.595829713,
        ],
    ];

    /// Oracle: residual after layer 0 at steps 0, 1, 2.
    const MODEL_LAYER_0: [[f64; 4]; 3] = [
        [-1.207416903, 1.04791293, -0.872629736, 1.939093993],
        [-1.103608972, 0.333276458, 2.272725154, -2.377150647],
        [2.080870238, 1.98374243, 2.213703574, -0.238326415],
    ];

    /// Oracle: residual after layer 1 at steps 0, 1, 2.
    const MODEL_LAYER_1: [[f64; 4]; 3] = [
        [-2.221141366, -0.165607986, -2.817519304, -1.005795096],
        [-2.838675144, 1.259745468, 3.540864342, 0.064112006],
        [0.797340489, 1.648453261, 2.325846653, 1.296087945],
    ];

    #[test]
    fn the_value_hash_matches_the_oracle() {
        // Printed by `q36_oracle.val` for tags 0, 7, 60 (the first values of each).
        assert_eq!(vals(0, 6), [0.375, -1.0, -0.375, 0.375, 0.875, -0.625]);
        assert_eq!(vals(7, 3), [1.0, -0.375, 0.25]);
        assert_eq!(vals(60, 3), [0.0, 0.75, -0.875]);
    }

    #[test]
    fn rms_plain_applies_the_weight_as_stored() {
        // x = [3, 4]: mean of squares 12.5, so x / sqrt(12.5) = [0.848528137, 1.131370850].
        // The weight [2, 0.5] multiplies as stored. A `1 + w` convention would give [3.0, 1.5] times that.
        let out = rms_plain(&[3.0, 4.0], &[2.0, 0.5], 0.0).unwrap();
        assert_close(&out, &[1.697056275, 0.565685425], "rms_plain");
        assert!(rms_plain(&[3.0, 4.0], &[2.0], 0.0).is_err());
    }

    #[test]
    fn qk_norm_scales_with_the_dimension_and_its_epsilon_acts_on_the_mean() {
        // D = 4 and x = 0.001 in every position: mean of squares 1e-6, so
        // rms_norm(x, None, 1e-6) = 0.001 / sqrt(2e-6) = 0.707106781 in every position.
        // q takes the factor 1/D = 0.25 and k takes 1/sqrt(D) = 0.5.
        // The epsilon added to the sum of squares instead of the mean would give 0.001 / sqrt(5e-6) = 0.447.
        let x = [0.001f32; 4];
        let (q, k) = qk_norm(&x, &x);
        assert_close(&q, &[0.176776695; 4], "q");
        assert_close(&k, &[0.353553391; 4], "k");
    }

    #[test]
    fn delta_matches_the_oracle_and_value_heads_follow_their_key_head() {
        let shape = shape(vec![AttentionKind::DeltaNet]);
        let fx = DeltaFx::new(0);
        let mut state = DeltaState::new(&shape.delta);
        let mut projector = DenseProjector::default();
        for (t, expected) in DELTA_OUT.iter().enumerate() {
            let x = vals(100 + t as u64, HIDDEN);
            let out = delta_step(&mut projector, &shape, &fx.weights(), &mut state, &x).unwrap();
            assert_close(&out, expected, &format!("delta step {t}"));
        }
        assert_close(state.rec(), &DELTA_STATE, "delta state");
    }

    #[test]
    fn the_delta_rule_matches_hand_values_over_two_steps() {
        // Two value rows of two key values. Row 0 starts at (1, 2) and row 1 at (3, 4).
        // Step 1: decay 1/2, beta 1/2, k (1, 0), q (0, 1), v (2, 4).
        //   decayed (1/2, 1), (3/2, 2); S k = (1/2, 3/2); delta = (3/4, 5/4);
        //   new rows (5/4, 1), (11/4, 2); output S q = (1, 2).
        // Step 2: decay 1/4, beta 1/2, k (1, 1), q (1, 2), v (1, 2).
        //   decayed (5/16, 1/4), (11/16, 1/2); S k = (9/16, 19/16); delta = (7/32, 13/32);
        //   new rows (17/32, 15/32), (35/32, 29/32); output S q = (47/32, 93/32).
        // Every value is a dyadic rational, so f32 holds it exactly.
        let mut rec = vec![1.0, 2.0, 3.0, 4.0];
        let out = delta_rule(
            &mut rec,
            (&[0.0, 1.0], &[1.0, 0.0], &[2.0, 4.0]),
            (0.5, 0.5),
        );
        assert_eq!(rec, [1.25, 1.0, 2.75, 2.0]);
        assert_eq!(out, [1.0, 2.0]);
        let out = delta_rule(
            &mut rec,
            (&[1.0, 2.0], &[1.0, 1.0], &[1.0, 2.0]),
            (0.25, 0.5),
        );
        assert_eq!(rec, [17.0 / 32.0, 15.0 / 32.0, 35.0 / 32.0, 29.0 / 32.0]);
        assert_eq!(out, [47.0 / 32.0, 93.0 / 32.0]);
    }

    #[test]
    fn delta_rejects_wrong_lengths() {
        let shape = shape(vec![AttentionKind::DeltaNet]);
        let fx = DeltaFx::new(0);
        let mut state = DeltaState::new(&shape.delta);
        let mut projector = DenseProjector::default();
        let short = delta_step(&mut projector, &shape, &fx.weights(), &mut state, &[0.0; 3]);
        assert!(matches!(short, Err(EngError::BadRequest(_))));
        assert_eq!(state, DeltaState::new(&shape.delta), "state unchanged");
    }

    #[test]
    fn attention_matches_the_oracle_over_three_positions() {
        let shape = shape(vec![AttentionKind::Full]);
        let fx = AttnFx::new(10);
        let mut cache = KvCache::default();
        let mut projector = DenseProjector::default();
        for (t, expected) in ATTN_OUT.iter().enumerate() {
            let x = vals(110 + t as u64, HIDDEN);
            let out = attn_step(
                &mut projector,
                &shape,
                &fx.weights(),
                &mut cache,
                t as u32,
                &x,
            )
            .unwrap();
            assert_close(&out, expected, &format!("attention position {t}"));
        }
        assert_eq!(cache.positions, 3);
        assert_eq!(cache.keys.len(), 3 * 2 * 4);
    }

    #[test]
    fn attention_with_a_full_rotary_dimension_depends_on_theta() {
        // With rotary_dim 4 the second rotation pair uses `theta^(-1/2)`, so theta changes the output
        // at positions 1 and 2. The oracle uses theta 100; theta 10_000 gives a different result.
        let outputs = |theta: f32| {
            let mut shape = shape(vec![AttentionKind::Full]);
            shape.attn.rotary_dim = 4;
            shape.attn.theta = theta;
            let fx = AttnFx::new(10);
            let mut cache = KvCache::default();
            let mut projector = DenseProjector::default();
            (0..3u32)
                .map(|t| {
                    let x = vals(110 + u64::from(t), HIDDEN);
                    attn_step(&mut projector, &shape, &fx.weights(), &mut cache, t, &x).unwrap()
                })
                .collect::<Vec<_>>()
        };
        for (t, (out, expected)) in outputs(100.0).iter().zip(&ATTN_OUT_ROT4).enumerate() {
            assert_close(out, expected, &format!("attention position {t}"));
        }
        let other = outputs(10_000.0);
        assert!(
            other[1]
                .iter()
                .zip(&ATTN_OUT_ROT4[1])
                .any(|(a, e)| (f64::from(*a) - e).abs() > 1e-2),
            "theta 10_000 must change the output at position 1"
        );
    }

    #[test]
    fn attention_rejects_a_position_that_skips_the_cache() {
        let shape = shape(vec![AttentionKind::Full]);
        let fx = AttnFx::new(10);
        let mut cache = KvCache::default();
        let mut projector = DenseProjector::default();
        let x = vals(110, HIDDEN);
        let skipped = attn_step(&mut projector, &shape, &fx.weights(), &mut cache, 1, &x);
        assert!(matches!(skipped, Err(EngError::BadRequest(_))));
        assert_eq!(cache, KvCache::default(), "cache unchanged");
    }

    #[test]
    fn moe_matches_the_oracle() {
        // The oracle routes the three inputs to experts [2, 3], [1, 0], and [3, 0]. The gap between
        // the last selected and the first rejected probability is at least 0.0069, far above f32 noise.
        let shape = shape(vec![AttentionKind::DeltaNet]);
        let fx = MoeFx::new(30);
        let mut projector = DenseProjector::default();
        for (t, expected) in MOE_OUT.iter().enumerate() {
            let x = vals(120 + t as u64, HIDDEN);
            let out = moe_step(&mut projector, &shape, &fx.weights(), &x).unwrap();
            assert_close(&out, expected, &format!("moe input {t}"));
        }
    }

    /// Owned weights of the two-layer model: a DeltaNet layer, then a dense attention layer.
    struct ModelFx {
        embed: DenseWeight,
        head: DenseWeight,
        norms: Vec<Vec<f32>>,
        delta: DeltaFx,
        attn: AttnFx,
        moes: [MoeFx; 2],
    }

    impl ModelFx {
        fn new() -> Self {
            Self {
                embed: matrix(50, 6, HIDDEN),
                head: matrix(56, 6, HIDDEN),
                norms: [51, 52, 53, 54, 55]
                    .iter()
                    .map(|t| norm(*t, HIDDEN))
                    .collect(),
                delta: DeltaFx::new(60),
                attn: AttnFx::new(80),
                moes: [MoeFx::new(70), MoeFx::new(90)],
            }
        }

        fn weights(&self) -> ModelW<'_, DenseWeight> {
            ModelW {
                embed: &self.embed,
                layers: vec![
                    LayerW {
                        input_norm: &self.norms[0],
                        post_norm: &self.norms[1],
                        mixer: MixerW::Delta(self.delta.weights()),
                        feed_forward: FeedForwardW::Moe(self.moes[0].weights()),
                    },
                    LayerW {
                        input_norm: &self.norms[2],
                        post_norm: &self.norms[3],
                        mixer: MixerW::Attn(self.attn.weights()),
                        feed_forward: FeedForwardW::Moe(self.moes[1].weights()),
                    },
                ],
                final_norm: &self.norms[4],
                lm_head: &self.head,
            }
        }
    }

    fn model_shape() -> HostShape {
        shape(vec![AttentionKind::DeltaNet, AttentionKind::Full])
    }

    #[test]
    fn the_model_step_matches_the_oracle_logits_and_layer_outputs() {
        let shape = model_shape();
        let fx = ModelFx::new();
        let weights = fx.weights();
        let mut state = HostState::new(&shape);
        let mut projector = DenseProjector::default();
        for (t, token) in [1u32, 4, 2].into_iter().enumerate() {
            let out = step(&mut projector, &shape, &weights, &mut state, token, true).unwrap();
            assert_close(&out.logits, &MODEL_LOGITS[t], &format!("logits step {t}"));
            assert_eq!(out.layer_outputs.len(), 2, "recorded layers");
            assert_close(
                &out.layer_outputs[0],
                &MODEL_LAYER_0[t],
                &format!("layer 0 step {t}"),
            );
            assert_close(
                &out.layer_outputs[1],
                &MODEL_LAYER_1[t],
                &format!("layer 1 step {t}"),
            );
            assert_eq!(state.pos(), t as u32 + 1);
        }
        assert!(matches!(state.layers()[0], LayerState::Delta(_)));
        assert!(matches!(state.layers()[1], LayerState::Attn(_)));
    }

    #[test]
    fn recording_is_optional_and_does_not_change_the_logits() {
        let shape = model_shape();
        let fx = ModelFx::new();
        let weights = fx.weights();
        let mut state = HostState::new(&shape);
        let mut projector = DenseProjector::default();
        let out = step(&mut projector, &shape, &weights, &mut state, 1, false).unwrap();
        assert!(out.layer_outputs.is_empty());
        assert_close(&out.logits, &MODEL_LOGITS[0], "logits");
    }

    #[test]
    fn a_bad_token_fails_the_step_and_invalidates_the_state() {
        let shape = model_shape();
        let fx = ModelFx::new();
        let weights = fx.weights();
        let mut state = HostState::new(&shape);
        let mut projector = DenseProjector::default();
        let bad = step(&mut projector, &shape, &weights, &mut state, 6, false);
        assert!(matches!(bad, Err(EngError::BadRequest(_))));
        let next = step(&mut projector, &shape, &weights, &mut state, 1, false);
        assert!(matches!(next, Err(EngError::BadRequest(message)) if message.contains("invalid")));
    }

    #[test]
    fn the_shape_follows_the_dense_specification() {
        let spec = &crate::family::spec::QWEN3_8_27B_DENSE;
        let shape = HostShape::from_spec(spec, 64).unwrap();
        assert_eq!(
            (shape.hidden, shape.vocab, shape.kinds.len()),
            (5120, 248_320, 64)
        );
        assert_eq!(
            (shape.attn.heads_q, shape.attn.heads_kv, shape.attn.head_dim),
            (24, 4, 256)
        );
        assert_eq!(shape.dense_inter, Some(17_408));
        assert_eq!(
            shape.delta,
            DeltaDims {
                key_heads: 16,
                key_dim: 128,
                value_heads: 48,
                value_dim: 128,
                conv_kernel: 4,
            }
        );
        assert_eq!(
            (
                shape.moe.experts,
                shape.moe.top_k,
                shape.moe.inter,
                shape.moe.shared_inter
            ),
            (0, 0, 0, 0)
        );
        let mut invalid = shape;
        invalid.dense_inter = Some(0);
        assert!(invalid.check().is_err());
    }

    #[test]
    fn the_shape_follows_the_stepping_stone_specification() {
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
        assert_eq!(shape.dense_inter, None);
        assert_eq!(
            (shape.hidden, shape.vocab, shape.kinds.len()),
            (2048, 248_320, 40)
        );
        let delta_layers = shape
            .kinds
            .iter()
            .filter(|k| **k == AttentionKind::DeltaNet)
            .count();
        assert_eq!(delta_layers, 30);
        // The specification gives theta 1e7 and epsilon 1e-6; the host shape must carry both unchanged.
        assert_eq!(shape.attn.theta, 10_000_000.0);
        assert_eq!(shape.eps, 1e-6);
        assert_eq!(shape.kinds[3], AttentionKind::Full);
        assert_eq!(
            shape.delta,
            DeltaDims {
                key_heads: 16,
                key_dim: 128,
                value_heads: 32,
                value_dim: 128,
                conv_kernel: 4
            }
        );
        assert_eq!(shape.delta.conv_dim(), 8192);
        assert_eq!(
            (
                shape.attn.heads_q,
                shape.attn.heads_kv,
                shape.attn.head_dim,
                shape.attn.rotary_dim
            ),
            (16, 2, 256, 64)
        );
        assert_eq!(
            shape.moe,
            MoeDims {
                experts: 256,
                top_k: 8,
                inter: 512,
                shared_inter: 512,
                renorm: true
            }
        );
        for bad in [0, 63, 258] {
            assert!(
                HostShape::from_spec(&QWEN3_6_35B_A3B, bad).is_err(),
                "rotary {bad}"
            );
        }
    }
}
