//! SenseNova Loop decode-step math, mirroring the vendor graph
//! (`sensenova-loop.cpp`, `qwen35.cpp`, `delta-net-base.cpp`).
//!
//! One token per call. The GDN core keeps the vendor's transposed state
//! layout (`mem[j * kd + i] = S[i][j]`), the attention layers carry a
//! growing K/V cache, and mHC layers route two residual streams through
//! Sinkhorn-normalized mixing matrices. Positions use text-mode MRoPE:
//! every section shares one position, so the rotation is a partial YaRN
//! rope over the first `rope.dims` dims of each head.

use crate::{
    EngError,
    family::math::{delta_rule, dot, expect_len, rms_plain, rms_unit, sigmoid, silu, softplus},
    family::project::{Projector, gguf::GgufMatrix as TensorRef},
};

use super::weights::{Layer, LoopShape, LoopWeights, Mixer};
use super::{LoopPlan, RopePlan, TRUNK_LAYERS};

impl Layer {
    pub(crate) fn delta_view(&self) -> DeltaView<'_, TensorRef> {
        let Mixer::Delta {
            qkvz,
            beta_alpha,
            out,
            conv,
            a,
            dt_bias,
            norm,
        } = &self.mixer
        else {
            unreachable!("delta view of a full-attention layer")
        };
        DeltaView {
            qkvz,
            beta_alpha,
            out,
            conv,
            a,
            dt_bias,
            norm,
        }
    }

    pub(crate) fn attn_view(&self) -> AttnView<'_, TensorRef> {
        let Mixer::Attn {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
        } = &self.mixer
        else {
            unreachable!("attention view of a recurrent layer")
        };
        AttnView {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
        }
    }

    pub(crate) fn ffn_views(&self) -> (&TensorRef, &TensorRef, &TensorRef) {
        (&self.ffn_gate, &self.ffn_up, &self.ffn_down)
    }
}

/// GDN view of one recurrent layer, generic over the weight handle.
pub(crate) struct DeltaView<'a, W> {
    pub(crate) qkvz: &'a W,
    pub(crate) beta_alpha: &'a W,
    pub(crate) out: &'a W,
    pub(crate) conv: &'a [f32],
    pub(crate) a: &'a [f32],
    pub(crate) dt_bias: &'a [f32],
    pub(crate) norm: &'a [f32],
}

/// Full-attention view of one layer, generic over the weight handle.
pub(crate) struct AttnView<'a, W> {
    pub(crate) q: &'a W,
    pub(crate) k: &'a W,
    pub(crate) v: &'a W,
    pub(crate) o: &'a W,
    pub(crate) q_norm: &'a [f32],
    pub(crate) k_norm: &'a [f32],
}

/// One mHC routing decision for a single token.
pub(crate) struct Route {
    /// Gated stream mix that replaces the residual input.
    pub(crate) input: Vec<f32>,
    /// Residual gains `2 * sigmoid(logit)` applied at the merge.
    pub(crate) post: [f32; 2],
    /// Sinkhorn-normalized mixing matrix entries `[m00 m01 m10 m11]`.
    pub(crate) mix: [f32; 4],
}

/// Per-logical-layer decode state.
pub(crate) struct LoopState {
    gdn_rec: Vec<Vec<f32>>,
    gdn_ring: Vec<Vec<f32>>,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
}

impl LoopState {
    pub(crate) fn new(shape: &LoopShape, plan: &LoopPlan) -> Self {
        let mut gdn_rec = Vec::new();
        let mut gdn_ring = Vec::new();
        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        // A logical visit owns its state even when it shares weights.
        for &recurrent in &plan.recurrent {
            if recurrent {
                gdn_rec.push(vec![
                    0.0;
                    shape.delta_value_heads
                        * shape.delta_key_dim
                        * shape.delta_value_dim
                ]);
                gdn_ring.push(vec![0.0; shape.conv_dim() * (shape.conv_kernel - 1)]);
            } else {
                gdn_rec.push(Vec::new());
                gdn_ring.push(Vec::new());
            }
            kv_k.push(Vec::new());
            kv_v.push(Vec::new());
        }
        Self {
            gdn_rec,
            gdn_ring,
            kv_k,
            kv_v,
        }
    }

    pub(crate) fn clear(&mut self) {
        for buf in self.gdn_rec.iter_mut().chain(&mut self.gdn_ring) {
            buf.fill(0.0);
        }
        for buf in self.kv_k.iter_mut().chain(&mut self.kv_v) {
            buf.clear();
        }
    }
}

/// Router logits through `[a || b]` after an unweighted RMS norm.
pub(crate) fn mhc_route<P: Projector>(
    projector: &mut P,
    shape: &LoopShape,
    router: &RouterView<'_, P::Weight>,
    fixed_identity: bool,
    pre_scale: f32,
    iterations: usize,
    streams: (&[f32], &[f32]),
) -> Result<Route, EngError> {
    let (a, b) = streams;
    expect_len(a, shape.hidden, "stream a")?;
    expect_len(b, shape.hidden, "stream b")?;
    let mut flat = Vec::with_capacity(2 * shape.hidden);
    flat.extend_from_slice(a);
    flat.extend_from_slice(b);
    let normed = rms_unit(&flat, 1e-6);
    let mut logits = projector.linear(router.weight, &normed)?;
    for (l, bias) in logits.iter_mut().zip(router.bias.iter()) {
        *l += bias;
    }
    let mut input = vec![0f32; shape.hidden];
    for j in 0..shape.hidden {
        input[j] = a[j] * sigmoid(logits[0]) + b[j] * sigmoid(logits[1]);
        input[j] *= pre_scale;
    }
    let route = Route {
        input,
        post: [2.0 * sigmoid(logits[2]), 2.0 * sigmoid(logits[3])],
        mix: if fixed_identity {
            [1.0, 0.0, 0.0, 1.0]
        } else {
            let mut m = [logits[4], logits[5], logits[6], logits[7]];
            let normalize = |m: &mut [f32; 4], i: usize, j: usize| {
                let delta = m[i] - m[j];
                m[i] = -softplus(-delta);
                m[j] = -softplus(delta);
            };
            for _ in 0..iterations {
                normalize(&mut m, 0, 1);
                normalize(&mut m, 2, 3);
                normalize(&mut m, 0, 2);
                normalize(&mut m, 1, 3);
            }
            m.map(f32::exp)
        },
    };
    Ok(route)
}

/// Applies one routing decision to the two streams with the branch output.
pub(crate) fn merge(a: &mut [f32], b: &mut [f32], branch: &[f32], route: &Route) {
    let mix = &route.mix;
    let next_a: Vec<f32> = (0..a.len())
        .map(|j| a[j] * mix[0] + b[j] * mix[1])
        .collect();
    let next_b: Vec<f32> = (0..a.len())
        .map(|j| a[j] * mix[2] + b[j] * mix[3])
        .collect();
    for j in 0..a.len() {
        a[j] = next_a[j] + branch[j] * route.post[0];
        b[j] = next_b[j] + branch[j] * route.post[1];
    }
}

/// Router weights as a generic view for tests.
pub(crate) struct RouterView<'a, W> {
    pub(crate) weight: &'a W,
    pub(crate) bias: &'a [f32],
}

/// The GDN `x / sqrt(sum(x^2) + eps)` norm of one head vector.
fn l2_head(x: &[f32], eps: f32) -> Vec<f32> {
    let scale = (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    x.iter().map(|v| v / scale).collect()
}

/// One gated-delta-net layer step: projection, causal conv, state update,
/// gated norm and the output projection. Returns the layer output.
pub(crate) fn delta_step<P: Projector>(
    projector: &mut P,
    shape: &LoopShape,
    view: &DeltaView<'_, P::Weight>,
    state: &mut LoopState,
    il: usize,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    expect_len(x, shape.hidden, "gdn input")?;
    let kh = shape.delta_key_heads;
    let kd = shape.delta_key_dim;
    let vh = shape.delta_value_heads;
    let vd = shape.delta_value_dim;
    let ck = shape.conv_kernel;
    let conv_dim = shape.conv_dim();

    let qkvz = projector.linear(view.qkvz, x)?;
    let ba = projector.linear(view.beta_alpha, x)?;
    let mut y = vec![0f32; conv_dim];
    {
        let qkv = &qkvz[..conv_dim];
        let ring = &mut state.gdn_ring[il];
        for c in 0..conv_dim {
            let taps = &view.conv[c * ck..(c + 1) * ck];
            let mut acc = taps[ck - 1] * qkv[c];
            for (t, past) in taps[..ck - 1]
                .iter()
                .zip(ring[c * (ck - 1)..(c + 1) * (ck - 1)].iter())
            {
                acc += t * past;
            }
            y[c] = silu(acc);
        }
        for (c, fresh) in qkv[..conv_dim].iter().enumerate() {
            let base = c * (ck - 1);
            ring.copy_within(base + 1..base + ck - 1, base);
            ring[base + ck - 2] = *fresh;
        }
    }

    let mut core = vec![0f32; shape.delta_inner()];
    let out_scale = 1.0 / (vd as f32).sqrt();
    for h in 0..vh {
        let key_head = h % kh;
        let q_raw = &y[key_head * kd..(key_head + 1) * kd];
        let k_raw = &y[(kh + key_head) * kd..(kh + key_head + 1) * kd];
        let v_raw = &y[(2 * kh) * kd + h * vd..(2 * kh) * kd + (h + 1) * vd];
        let q = l2_head(q_raw, shape.eps);
        let k = l2_head(k_raw, shape.eps);
        let beta = sigmoid(ba[h]);
        let decay = (view.a[h] * softplus(ba[vh + h] + view.dt_bias[h])).exp();

        let mem = &mut state.gdn_rec[il][h * kd * vd..(h + 1) * kd * vd];
        let mut out = delta_rule(mem, (&q, &k, v_raw), (decay, beta));
        for value in &mut out {
            *value *= out_scale;
        }
        let gated = rms_plain(&out, view.norm, shape.eps)?;
        let z = &qkvz[conv_dim + h * vd..conv_dim + (h + 1) * vd];
        for (o, (g, z_i)) in core[h * vd..(h + 1) * vd]
            .iter_mut()
            .zip(gated.iter().zip(z))
        {
            *o = g * silu(*z_i);
        }
    }
    projector.linear(view.out, &core)
}

/// One full-attention layer step with the packed `[query | gate]` q
/// projection, partial YaRN rope over the leading rotary dims, GQA over
/// the cached keys and values, and the sigmoid output gate. The token's
/// own key and value join the cache before scoring.
pub(crate) fn attn_step<P: Projector>(
    projector: &mut P,
    shape: &LoopShape,
    rope: &RopePlan,
    view: &AttnView<'_, P::Weight>,
    state: &mut LoopState,
    il: usize,
    tok: (&[f32], u32),
) -> Result<Vec<f32>, EngError> {
    let (x, pos) = tok;
    expect_len(x, shape.hidden, "attention input")?;
    let hd = shape.head_dim;
    let q_full = projector.linear(view.q, x)?;
    let k = projector.linear(view.k, x)?;
    let v = projector.linear(view.v, x)?;

    let mut keys = vec![0f32; shape.heads_kv * hd];
    for kv_head in 0..shape.heads_kv {
        let mut key = rms_plain(&k[kv_head * hd..(kv_head + 1) * hd], view.k_norm, shape.eps)?;
        apply_yarn(&mut key, rope, pos);
        keys[kv_head * hd..(kv_head + 1) * hd].copy_from_slice(&key);
    }
    let kv_len = state.kv_k[il].len();
    state.kv_k[il].extend_from_slice(&keys);
    state.kv_v[il].extend_from_slice(&v);

    let scores_scale = 1.0 / (hd as f32).sqrt();
    let mut scores = Vec::with_capacity(kv_len / (hd * shape.heads_kv) + 1);
    let mut attn_out = vec![0f32; shape.heads_q * hd];
    for h in 0..shape.heads_q {
        let base = h * 2 * hd;
        let mut q = rms_plain(&q_full[base..base + hd], view.q_norm, shape.eps)?;
        apply_yarn(&mut q, rope, pos);
        let kv_head = h / (shape.heads_q / shape.heads_kv);
        for key in state.kv_k[il].chunks(shape.heads_kv * hd) {
            let head = &key[kv_head * hd..(kv_head + 1) * hd];
            scores.push(dot(&q, head)? * scores_scale);
        }
        softmax_in_place(&mut scores);
        let values = &state.kv_v[il];
        let mut acc = vec![0f32; hd];
        for (t, weight) in scores.iter().enumerate() {
            let head = &values[t * shape.heads_kv * hd + kv_head * hd..][..hd];
            for (a, v_i) in acc.iter_mut().zip(head) {
                *a += weight * v_i;
            }
        }
        let gate = &q_full[base + hd..base + 2 * hd];
        for j in 0..hd {
            attn_out[h * hd + j] = acc[j] * sigmoid(gate[j]);
        }
        scores.clear();
    }
    projector.linear(view.o, &attn_out)
}

/// Softmax in place over `scores`.
fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

/// Partial YaRN rope over the leading `rope.dims` values, NEOX pairing
/// `(i, i + dims/2)`. Fast pairs keep the original frequency; slow pairs
/// stretch by the YaRN factor. The vendor context cancels the kernel's mscale.
pub(crate) fn apply_yarn(head: &mut [f32], rope: &RopePlan, pos: u32) {
    let dims = rope.dims;
    let half = dims / 2;
    let base = rope.freq_base;
    let freq_scale = 1.0 / rope.factor;
    let low = corr_dim(dims, rope.original_context as f32, 32.0, base)
        .floor()
        .max(0.0);
    let high = corr_dim(dims, rope.original_context as f32, 1.0, base)
        .ceil()
        .min(dims as f32 - 1.0);
    let mut theta = pos as f32;
    let theta_scale = base.powf(-2.0 / dims as f32);
    for i in 0..half {
        let ramp = ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
        let angle = theta * (1.0 - ramp) + theta * freq_scale * ramp;
        let (sin, cos) = angle.sin_cos();
        let a = head[i];
        let b = head[i + half];
        head[i] = a * cos - b * sin;
        head[i + half] = b * cos + a * sin;
        theta *= theta_scale;
    }
}

/// The per-pair inverse frequencies of the YaRN rope as a table: the same
/// progressive powers and the same ramp blend [`apply_yarn`] computes inline,
/// so a device table built here agrees with the host reference by construction
/// (the kernel reads `pos * table[i]` where the host applies `theta * blend`).
#[cfg_attr(
    all(not(test), not(feature = "native-cuda")),
    expect(
        dead_code,
        reason = "read by the native-cuda step and the family tests"
    )
)]
pub(crate) fn yarn_inv_freq(rope: &RopePlan) -> Vec<f32> {
    let dims = rope.dims;
    let half = dims / 2;
    let base = rope.freq_base;
    let freq_scale = 1.0 / rope.factor;
    let low = corr_dim(dims, rope.original_context as f32, 32.0, base)
        .floor()
        .max(0.0);
    let high = corr_dim(dims, rope.original_context as f32, 1.0, base)
        .ceil()
        .min(dims as f32 - 1.0);
    let mut theta = 1.0f32;
    let theta_scale = base.powf(-2.0 / dims as f32);
    let mut table = Vec::with_capacity(half);
    for i in 0..half {
        let ramp = ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
        table.push(theta * ((1.0 - ramp) + freq_scale * ramp));
        theta *= theta_scale;
    }
    table
}

/// YaRN correction boundary: `dims * ln(orig / (beta * 2pi)) / (2 ln base)` —
/// the dim index where a `beta`-wavelength fits the original context.
/// `low` clamps at 0 and `high` at `dims - 1`, as in `ggml_rope_yarn_corr_dims`.
fn corr_dim(dims: usize, orig: f32, beta: f32, base: f32) -> f32 {
    dims as f32 * (orig / (beta * std::f32::consts::TAU)).ln() / (2.0 * base.ln())
}

pub(crate) fn logits<P: Projector<Weight = TensorRef>>(
    projector: &mut P,
    weights: &LoopWeights,
    hidden: &[f32],
) -> Result<Vec<f32>, EngError> {
    let normed = rms_plain(hidden, weights.output_norm(), weights.shape().eps)?;
    projector.linear(weights.output(), &normed)
}

/// Embeds a token and feeds all 56 logical visits, leaving the head to the caller. Any
/// projector over the weights: the parameter exists so a test can emulate the quantized model.
pub(crate) fn trunk<P: Projector<Weight = TensorRef>>(
    projector: &mut P,
    weights: &LoopWeights,
    state: &mut LoopState,
    token: u32,
    pos: u32,
) -> Result<Vec<f32>, EngError> {
    trunk_inner(projector, weights, state, token, pos, None)
}

/// The parity rig's twin: also snapshots the residual row after every visit.
#[cfg_attr(
    not(all(test, feature = "native-cuda", target_os = "linux")),
    expect(dead_code, reason = "used by the CUDA device parity rig")
)]
pub(crate) fn trunk_traced<P: Projector<Weight = TensorRef>>(
    projector: &mut P,
    weights: &LoopWeights,
    state: &mut LoopState,
    token: u32,
    pos: u32,
    trace: &mut Vec<Vec<f32>>,
) -> Result<Vec<f32>, EngError> {
    trunk_inner(projector, weights, state, token, pos, Some(trace))
}

fn trunk_inner<P: Projector<Weight = TensorRef>>(
    projector: &mut P,
    weights: &LoopWeights,
    state: &mut LoopState,
    token: u32,
    pos: u32,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<Vec<f32>, EngError> {
    let shape = weights.shape();
    let plan = weights.plan();
    let rope = weights.rope();
    let mut cur = projector.rows(weights.token_embd(), &[token])?;
    let mut a: Vec<f32> = Vec::new();
    let mut b: Vec<f32> = Vec::new();

    for il in 0..TRUNK_LAYERS {
        let physical = plan.physical(il);
        let layer = &weights.layers()[physical];
        let mhc = plan.mhc_active(il);
        if mhc && physical == plan.mhc_start {
            a = cur.clone();
            b = cur.clone();
        }
        let attn_route = if mhc {
            let router = RouterView {
                weight: &layer.mhc_attn.as_ref().expect("mhc layer").weight,
                bias: &layer.mhc_attn.as_ref().expect("mhc layer").bias,
            };
            Some(mhc_route(
                projector,
                shape,
                &router,
                plan.mhc_fixed_identity,
                plan.mhc_pre_scale,
                plan.mhc_iterations,
                (&a, &b),
            )?)
        } else {
            None
        };
        let normed = rms_plain(
            attn_route
                .as_ref()
                .map_or(cur.as_slice(), |r| r.input.as_slice()),
            &layer.attn_norm,
            shape.eps,
        )?;

        let branch = match &layer.mixer {
            super::weights::Mixer::Delta { .. } => {
                let view = layer.delta_view();
                delta_step(projector, shape, &view, state, il, &normed)?
            }
            super::weights::Mixer::Attn { .. } => {
                let view = layer.attn_view();
                attn_step(projector, shape, rope, &view, state, il, (&normed, pos))?
            }
        };
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(branch.clone());
        }

        if mhc {
            merge(
                &mut a,
                &mut b,
                &branch,
                attn_route.as_ref().expect("mhc layer"),
            );
            let router = RouterView {
                weight: &layer.mhc_ffn.as_ref().expect("mhc layer").weight,
                bias: &layer.mhc_ffn.as_ref().expect("mhc layer").bias,
            };
            let ffn_route = mhc_route(
                projector,
                shape,
                &router,
                plan.mhc_fixed_identity,
                plan.mhc_pre_scale,
                plan.mhc_iterations,
                (&a, &b),
            )?;
            let ffn_norm = rms_plain(&ffn_route.input, &layer.post_norm, shape.eps)?;
            let ffn_out = ffn(projector, layer, shape, &ffn_norm)?;
            merge(&mut a, &mut b, &ffn_out, &ffn_route);
            cur = (0..shape.hidden).map(|j| 0.5 * (a[j] + b[j])).collect();
        } else {
            for (c, add) in cur.iter_mut().zip(&branch) {
                *c += add;
            }
            let ffn_norm = rms_plain(&cur, &layer.post_norm, shape.eps)?;
            let ffn_out = ffn(projector, layer, shape, &ffn_norm)?;
            for (c, add) in cur.iter_mut().zip(&ffn_out) {
                *c += add;
            }
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(cur.clone());
        }
    }

    Ok(cur)
}

fn ffn<P: Projector<Weight = TensorRef>>(
    projector: &mut P,
    layer: &super::weights::Layer,
    shape: &LoopShape,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let (gate_w, up_w, down_w) = layer.ffn_views();
    let up = projector.linear(up_w, x)?;
    let gate = projector.linear(gate_w, x)?;
    let mut hidden = Vec::with_capacity(shape.feed_forward);
    for (u, g) in up.iter().zip(&gate) {
        hidden.push(u * silu(*g));
    }
    projector.linear(down_w, &hidden)
}
