//! Host reference forward pass for Raon-1.
//!
//! The full 25-layer F32 graph in the fork's graph order (KRAFTON fork
//! `fb895662`, `src/models/raon1.cpp`), reading dequantized tensors from the
//! checkpoint opener. Every stage the fork's eval-callback names is emitted to
//! a sink in ggml linear order (fastest dim first), so the fixture parity test
//! can compare against the frozen `fixtures1` dumps.
//!
//! Numeric scheme, stated exactly: dot products and the norm sums accumulate
//! in F64 and store F32; every other elementwise fold stays F32. Before each
//! quantized dot the activation is roundtrip-quantized to the weight's
//! vec-dot type (Q8_K blocks for the K-quants, Q8_0 blocks for the plain
//! quants, an F16 roundtrip for F16 weights, nothing for F32), reproducing
//! the fork's CPU mul_mat. The F64 accumulation is a deliberate divergence:
//! the fork's F32 dot carries ~1e-6-of-range noise, and the exact value is
//! the better oracle for the fixture tolerances in `tests.rs`.
//!
//! This is an oracle, not a serving path: plain scalar loops, no tiling, no
//! kernels. Math contracts that are easy to get wrong are commented where they
//! are used.

use crate::EngError;
use crate::family::raon::raon1::{Raon1Model, Raon1Spec};
use crate::weights::GgufType;
use crate::weights::MetaValue;

/// Hyperparameters the fork's raon1 loader reads from the checkpoint header.
/// Read, not hardcoded: the opener pins geometry only.
pub struct Hyper {
    pub streams: usize,
    pub norm_eps: f32,
    pub latent_eps: f32,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub used_experts: usize,
    pub weights_norm: bool,
    pub gate_lower_bound: f32,
    pub rope_dim: usize,
}

/// The smallest weight sum `weights_norm` divides by, clamped the way ggml
/// does (smallest positive normal F16).
const WEIGHT_SUM_MIN: f32 = 6.103515_6e-5; // 2^-14, smallest positive normal F16

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("raon1 reference: {}", message.into()))
}

fn meta_f32(model: &Raon1Model, key: &str) -> Result<f32, EngError> {
    match model.header().meta(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        _ => Err(reject(format!("metadata key {key} is missing or not f32"))),
    }
}

fn meta_u32(model: &Raon1Model, key: &str) -> Result<u32, EngError> {
    model
        .header()
        .meta_u32(key)
        .ok_or_else(|| reject(format!("metadata key {key} is missing or not u32")))
}

fn meta_bool(model: &Raon1Model, key: &str) -> Result<bool, EngError> {
    match model.header().meta(key) {
        Some(MetaValue::Bool(v)) => Ok(*v),
        _ => Err(reject(format!("metadata key {key} is missing or not bool"))),
    }
}

/// Reads the Raon-1 hyperparameters the reference needs, rejecting values the
/// reference does not implement (gating func 2 = sigmoid, weight scale 1.0).
pub fn hyper(model: &Raon1Model, spec: Raon1Spec) -> Result<Hyper, EngError> {
    if meta_u32(model, "raon1.expert_gating_func")? != 2 {
        return Err(reject("expert_gating_func is not 2 (sigmoid)"));
    }
    let scale = meta_f32(model, "raon1.expert_weights_scale")?;
    if scale != 1.0 {
        return Err(reject(format!("expert_weights_scale {scale} is not 1.0")));
    }
    let rope_dim = meta_u32(model, "raon1.rope.dimension_count")? as usize;
    // Geometry the MLA path indexes silently if it disagrees: the shared KV
    // projection must be exactly the latent plus the rope dims, and the
    // per-head absorbed key width the key width minus the rope part.
    if u64::try_from(rope_dim).is_err() || spec.kv_a_width != spec.kv_latent + rope_dim as u64 {
        return Err(reject(format!(
            "rope.dimension_count {rope_dim} disagrees with kv_a_width {} minus kv_latent {}",
            spec.kv_a_width, spec.kv_latent
        )));
    }
    if spec.k_head != spec.key_mla - rope_dim as u64 {
        return Err(reject(format!(
            "k_head {} disagrees with key_mla {} minus rope_dim {rope_dim}",
            spec.k_head, spec.key_mla
        )));
    }
    // hc_site reads scale[0..=3]; fewer streams would slice out of range.
    if (spec.streams as usize) < 4 {
        return Err(reject(format!(
            "hyper_connection.count {} is below the 4 streams hc_site indexes",
            spec.streams
        )));
    }
    Ok(Hyper {
        streams: spec.streams as usize,
        norm_eps: meta_f32(model, "raon1.attention.layer_norm_rms_epsilon")?,
        latent_eps: meta_f32(model, "raon1.attention.latent_norm_rms_epsilon")?,
        hc_eps: meta_f32(model, "raon1.hyper_connection.epsilon")?,
        hc_magnitude: meta_f32(model, "raon1.hyper_connection.magnitude")?,
        situ_beta: meta_f32(model, "raon1.activation.situ_beta")?,
        situ_linear_beta: meta_f32(model, "raon1.activation.situ_linear_beta")?,
        used_experts: meta_u32(model, "raon1.expert_used_count")? as usize,
        weights_norm: meta_bool(model, "raon1.expert_weights_norm")?,
        gate_lower_bound: meta_f32(model, "raon1.kda.gate_lower_bound")?,
        rope_dim: meta_u32(model, "raon1.rope.dimension_count")? as usize,
    })
}

/// `y[o, t] = sum_k w[o, k] * x[k, t]`, ggml layout (fastest dim first) for
/// both operands and result. This is `ggml_mul_mat(w, x)` in the fork.
/// Accumulation is F64: the fork's F32 SIMD dot carries ~1e-6-of-range noise
/// on cancellation-heavy rows, and the exact value is the better oracle.
fn mm(w: &[f32], x: &[f32], k: usize, n: usize, nt: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * nt];
    for t in 0..nt {
        let xc = &x[t * k..(t + 1) * k];
        for o in 0..n {
            let wrow = &w[o * k..(o + 1) * k];
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += wrow[i] as f64 * xc[i] as f64;
            }
            y[o + t * n] = acc as f32;
        }
    }
    y
}

/// Single-column `mm` (one token), the MoE path: `out[o] = w[o, :] . x`.
fn mv(w: &[f32], x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (o, slot) in out.iter_mut().enumerate() {
        let wrow = &w[o * k..(o + 1) * k];
        let mut acc = 0.0f64;
        for i in 0..k {
            acc += wrow[i] as f64 * x[i] as f64;
        }
        *slot = acc as f32;
    }
    out
}

/// `nearest_int` as ggml-quants.c defines it: the 12582912.0f bit trick,
/// round-half-even in the mantissa it lands on.
fn ggml_nearest_int(f: f32) -> i32 {
    (f32::to_bits(f + 12582912.0) & 0x007f_ffff) as i32 - 0x0040_0000
}

/// f32 -> f16 -> f32 through the `half` crate (round-to-nearest-even), the
/// rounding `block_q8_0` applies to its scale and the F16 vec-dot applies to
/// an F32 activation.
pub(crate) fn f16_roundtrip(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// How the host reference rounds an activation before a quantized dot.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActivationRule {
    /// The fork's CPU rule: Q8_K blocks of 256 for the K-quants, Q8_0 blocks of 32 for the plain
    /// quants.
    ForkCpu,
    /// The device rule: blocks of 32 for every quantized weight, scale `amax / 127` stored as
    /// f16, codes rounded from the unrounded scale.
    Q8_1Blocks,
    /// No rounding: every dot reads the f32 activation. The reference the rounding rules are
    /// measured against.
    Exact,
}

#[cfg(test)]
thread_local! {
    static ACTIVATION_RULE: std::cell::Cell<ActivationRule> =
        const { std::cell::Cell::new(ActivationRule::ForkCpu) };
}

/// Runs `f` with the host reference rounding activations by `rule` on this thread.
#[cfg(all(test, feature = "native-cuda"))]
pub(crate) fn with_activation_rule<R>(rule: ActivationRule, f: impl FnOnce() -> R) -> R {
    let previous = ACTIVATION_RULE.with(|cell| cell.replace(rule));
    let result = f();
    ACTIVATION_RULE.with(|cell| cell.set(previous));
    result
}

/// The device activation rounding: per block of 32, `d = amax / 127`, `q = round(x / d)`, the
/// value read back as `q * f16(d)`.
#[cfg(test)]
fn quantize_q8_1_blocks(x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len() % 32, 0, "blocks of 32 need whole blocks");
    let mut out = x.to_vec();
    for blk in out.as_chunks_mut::<32>().0 {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            continue;
        }
        let d = amax / 127.0;
        let stored = f16_roundtrip(d);
        for v in blk.iter_mut() {
            *v = stored * (*v / d).round();
        }
    }
    out
}

/// Roundtrip-quantizes one contraction-length activation column the way the
/// fork's CPU mul_mat does before a quantized dot: Q8_K blocks of 256 for the
/// K-quants, Q8_0 blocks of 32 for the plain quants, an F16 roundtrip for F16
/// weights (the CPU path's vec_dot_type for F16 is F16). Q8_K follows
/// `quantize_row_q8_K_ref`; Q8_0 follows `quantize_row_q8_0_ref`, including
/// its detail that the codes use the reciprocal of the unrounded scale while
/// the stored (f16-rounded) scale dequantizes them.
fn quantize_activation_column(ty: GgufType, x: &[f32]) -> Vec<f32> {
    #[cfg(test)]
    if ACTIVATION_RULE.with(std::cell::Cell::get) == ActivationRule::Exact {
        return x.to_vec();
    }
    #[cfg(test)]
    if ACTIVATION_RULE.with(std::cell::Cell::get) == ActivationRule::Q8_1Blocks
        && !matches!(ty, GgufType::F16 | GgufType::F32)
    {
        return quantize_q8_1_blocks(x);
    }
    let mut out = match ty {
        GgufType::F16 => x.iter().map(|&v| f16_roundtrip(v)).collect(),
        _ => x.to_vec(),
    };
    match ty {
        GgufType::Q2K | GgufType::Q3K | GgufType::Q4K | GgufType::Q5K | GgufType::Q6K => {
            assert_eq!(x.len() % 256, 0, "Q8_K needs whole 256 blocks");
            let qkk = 256; // Q8_K block size
            for blk in out.chunks_exact_mut(qkk) {
                let (mut max, mut amax) = (0.0f32, 0.0f32);
                for &v in blk.iter() {
                    let ax = v.abs();
                    if ax > amax {
                        amax = ax;
                        max = v;
                    }
                }
                if amax == 0.0 {
                    blk.fill(0.0);
                    continue;
                }
                let iscale = -127.0 / max;
                for v in blk.iter_mut() {
                    let q = ggml_nearest_int(iscale * *v).min(127);
                    *v = q as f32 / iscale;
                }
            }
        }
        GgufType::Q40 | GgufType::Q41 | GgufType::Q50 | GgufType::Q51 | GgufType::Q80 => {
            assert_eq!(x.len() % 32, 0, "Q8_0 needs whole 32 blocks");
            let qk8 = 32; // Q8_0 block size
            for blk in out.chunks_exact_mut(qk8) {
                let mut amax = 0.0f32;
                for &v in blk.iter() {
                    amax = amax.max(v.abs());
                }
                let d = amax / 127.0;
                if d == 0.0 {
                    blk.fill(0.0);
                    continue;
                }
                // ggml takes the reciprocal of the unrounded d for the codes
                // and rounds d to f16 only for the stored scale; taking it
                // from the rounded scale mis-rounds borderline elements by
                // one code step.
                let dr = f16_roundtrip(d);
                let id = 1.0 / d;
                for v in blk.iter_mut() {
                    *v = dr * (*v * id).round();
                }
            }
        }
        _ => {}
    }
    out
}

/// `mm` against a checkpoint tensor, reproducing the fork's CPU mul_mat
/// numerics: the activation is roundtrip-quantized to the weight's vec-dot
/// type first. Measured against the fork's `fixtures1` stage dumps: the
/// quantized dot matches the fork to ~1e-7 of stage range on Q6_K weights
/// (`the_dense_ffn_gate_element_pins_the_quantized_dot` in `tests.rs`).
pub(crate) fn fork_mm(
    model: &Raon1Model,
    name: &str,
    x: &[f32],
    k: usize,
    n: usize,
    nt: usize,
) -> Result<Vec<f32>, EngError> {
    let (w, ty) = model.read_f32_typed(name)?;
    let mut a = Vec::with_capacity(x.len());
    for t in 0..nt {
        let col = &x[t * k..(t + 1) * k];
        a.extend(quantize_activation_column(ty, col));
    }
    Ok(mm(&w, &a, k, n, nt))
}

/// Single-token `fork_mm` against an already-read expert weight slice.
pub(crate) fn fork_mv(w: &[f32], ty: GgufType, x: &[f32], k: usize, n: usize) -> Vec<f32> {
    let a = quantize_activation_column(ty, x);
    mv(w, &a, k, n)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    // ggml's softplus: logf(1 + expf(x)), overflow semantics included: x > 88
    // overflows to inf, which propagates through the RDA fold as NaN. Kept
    // for parity; a numerically stable form belongs to the kernel port, not
    // to this oracle.
    (1.0 + x.exp()).ln()
}

/// SiTU: `tanh(gate/beta)*beta * sigmoid(gate)`, then the up projection
/// passes through `tanh(up/lb)*lb` when `lb > 0` (fork `build_situ`).
fn situ(gate: &[f32], up: &[f32], beta: f32, lb: f32) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let act = (g / beta).tanh() * beta * sigmoid(g);
            let upq = if lb > 0.0 { (u / lb).tanh() * lb } else { u };
            act * upq
        })
        .collect()
}

/// `x / sqrt(mean(x^2) + eps)` over columns of `d` elements, scaled by `w`
/// (`ggml_rms_norm` + mul, fork `build_norm` RMS).
fn rms_norm_scaled(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for (col, yc) in y.chunks_exact_mut(d).enumerate() {
        let xc = &x[col * d..(col + 1) * d];
        let mut sum = 0.0f64;
        for &v in xc {
            sum += v as f64 * v as f64;
        }
        let scale = 1.0 / (sum / d as f64 + eps as f64).sqrt() as f32;
        for (slot, (&v, &wv)) in yc.iter_mut().zip(xc.iter().zip(w)) {
            *slot = v * scale * wv;
        }
    }
    y
}

/// `x / max(||x||, eps)` over columns of `d` elements (fork `ggml_l2_norm`).
fn l2_norm_cols(x: &mut [f32], d: usize, eps: f32) {
    for col in x.chunks_exact_mut(d) {
        let mut sum = 0.0f64;
        for &v in &*col {
            sum += v as f64 * v as f64;
        }
        let scale = 1.0 / sum.sqrt().max(eps as f64) as f32;
        for slot in col.iter_mut() {
            *slot *= scale;
        }
    }
}

/// One mHC site: the mixes/pre/post/comb computation over the full residual.
struct HcSite {
    branch: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn hc_site(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    inp: &[f32],
    layer: usize,
    site: &str,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<HcSite, EngError> {
    let h = spec.hidden as usize;
    let hc = hy.streams;
    let nm = hc - 1;
    let nt = inp.len() / (h * hc);

    // rms over the flattened [hidden*hc, nt] view; inp is already that layout.
    let mut rms = vec![0.0f32; nt];
    for t in 0..nt {
        let mut sum = 0.0f64;
        for r in 0..h * hc {
            let v = inp[r + t * h * hc] as f64;
            sum += v * v;
        }
        rms[t] = (sum / (h * hc) as f64 + hy.hc_eps as f64).sqrt() as f32;
    }

    let mix = spec.mix_count() as usize;
    let fn_w = model.read_f32(&format!("blk.{layer}.hc_{site}_fn.weight"))?;
    let mut mixes = vec![0.0f32; mix * nt];
    for t in 0..nt {
        for m in 0..mix {
            let mut acc = 0.0f64;
            for r in 0..h * hc {
                acc += fn_w[m * h * hc + r] as f64 * inp[r + t * h * hc] as f64;
            }
            mixes[m + t * mix] = (acc / rms[t] as f64) as f32;
        }
    }
    emit(&format!("hc_mixes-{layer}"), &mixes);

    let scale = model.read_f32(&format!("blk.{layer}.hc_{site}_scale.weight"))?;
    let base = model.read_f32(&format!("blk.{layer}.hc_{site}_base.weight"))?;

    let affine = |off: usize, si: usize, m: usize, t: usize| {
        mixes[(off + m) + t * mix] * scale[si] + base[off + m]
    };
    let mut pre = vec![0.0f32; hc * nt];
    let mut post = vec![0.0f32; hc * nt];
    for t in 0..nt {
        for s in 0..hc {
            pre[s + t * hc] = sigmoid(affine(0, 0, s, t));
            post[s + t * hc] = hy.hc_magnitude * sigmoid(affine(hc, 1, s, t));
        }
    }
    // hc_post/hc_comb are emitted by the caller after the mixer: the fork's
    // graph orders them there, not beside hc_pre.
    emit(&format!("hc_pre-{layer}"), &pre);

    // Helmert rows: H[i][j] over [hc, nm]. The fork chains two ggml
    // matmuls; net effect e[i][x] = sum_jq H[x][q] * z[j][q] * H[i][j],
    // measured against the fork's asymmetric layer-2 hc_comb fixture.
    let mut helmert = vec![vec![0.0f32; nm]; hc];
    for (i, row) in helmert.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let v = if i <= j {
                1.0
            } else if i == j + 1 {
                -((j + 1) as f32)
            } else {
                0.0
            };
            *slot = v / (((j + 1) * (j + 2)) as f32).sqrt();
        }
    }

    // chart_base = 0.5 * I + 0.5 / hc elementwise (off-diagonal 0.5/hc).
    let mut comb = vec![0.0f32; hc * hc * nt];
    for t in 0..nt {
        // z as a matrix: Z[j, q] at flat q + j*nm.
        let mut z = vec![0.0f32; nm * nm];
        for j in 0..nm {
            for q in 0..nm {
                z[j * nm + q] = affine(2 * hc, 2, j * nm + q, t);
            }
        }
        let mut e = vec![vec![0.0f32; hc]; hc];
        let mut rho = 0.0f32;
        for i in 0..hc {
            for x in 0..hc {
                let mut acc = 0.0f32;
                for j in 0..nm {
                    for q in 0..nm {
                        acc += helmert[x][q] * z[j * nm + q] * helmert[i][j];
                    }
                }
                e[i][x] = acc;
                let chart = 0.5 * ((i == x) as u8 as f32) + 0.5 / hc as f32;
                rho = rho.max(-e[i][x] / chart);
            }
        }
        rho = 1.0 + rho.max(0.0);
        let gamma = sigmoid(scale[3]);
        for i in 0..hc {
            for x in 0..hc {
                let phi = e[i][x] / rho + 0.5 * ((i == x) as u8 as f32) + 0.5 / hc as f32;
                let ident = (i == x) as u8 as f32;
                comb[i + x * hc + t * hc * hc] = ident + gamma * (phi - ident);
            }
        }
    }

    // Unfused hc_pre: sum the streams weighted by pre.
    let mut branch = vec![0.0f32; h * nt];
    for t in 0..nt {
        for k in 0..h {
            let mut acc = 0.0f32;
            for s in 0..hc {
                acc += inp[k + s * h + t * h * hc] * pre[s + t * hc];
            }
            branch[k + t * h] = acc;
        }
    }
    Ok(HcSite { branch, post, comb })
}

/// Unfused hc_post: mix streams through comb, add the gated branch, concat
/// streams back into the residual `[hidden, hc, nt]`.
fn hc_post_apply(
    inp: &[f32],
    branch: &[f32],
    site: &HcSite,
    hy: &Hyper,
    spec: Raon1Spec,
) -> Vec<f32> {
    let h = spec.hidden as usize;
    let hc = hy.streams;
    let nt = branch.len() / h;
    let mut out = vec![0.0f32; h * hc * nt];
    for t in 0..nt {
        for dst in 0..hc {
            for k in 0..h {
                let mut acc = 0.0f32;
                for src in 0..hc {
                    acc += inp[k + src * h + t * h * hc] * site.comb[dst + src * hc + t * hc * hc];
                }
                acc += branch[k + t * h] * site.post[dst + t * hc];
                out[k + dst * h + t * h * hc] = acc;
            }
        }
    }
    out
}

/// Causal conv1d over a zeroed 3-tap state prefix, then SiLU: the fork's
/// `build_rda_conv` for one q/k/v third. `x` is `[d_inner, nt]`.
fn rda_conv(
    model: &Raon1Model,
    layer: usize,
    third: &str,
    x: &[f32],
    din: usize,
) -> Result<Vec<f32>, EngError> {
    let nt = x.len() / din;
    let w = model.read_f32(&format!("blk.{layer}.ssm_conv1d_{third}.weight"))?;
    let mut y = vec![0.0f32; x.len()];
    for c in 0..din {
        // Weight layout [conv_kernel, 1, d_inner]: tap j of channel c at c*4 + j.
        let taps = [w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]];
        for t in 0..nt {
            let tap = |back: usize| -> f32 {
                if t < back {
                    0.0
                } else {
                    x[c + (t - back) * din]
                }
            };
            let acc = taps[0] * tap(3) + taps[1] * tap(2) + taps[2] * tap(1) + taps[3] * tap(0);
            y[c + t * din] = silu(acc);
        }
    }
    Ok(y)
}

/// The RDA (gated delta-rule) layer with KRAFTON's per-channel decay.
#[allow(clippy::too_many_arguments)]
fn rda_layer(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    cur: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let heads = spec.heads as usize;
    let hd = spec.head_dim as usize;
    let din = heads * hd;
    let nt = cur.len() / h;

    let w = |name: &str| model.read_f32(&format!("blk.{layer}.{name}"));
    let xq = fork_mm(
        model,
        &format!("blk.{layer}.attn_q.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let xk = fork_mm(
        model,
        &format!("blk.{layer}.attn_k.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let xv = fork_mm(
        model,
        &format!("blk.{layer}.attn_v.weight"),
        cur,
        h,
        din,
        nt,
    )?;
    let mut qc = rda_conv(model, layer, "q", &xq, din)?;
    let mut kc = rda_conv(model, layer, "k", &xk, din)?;
    let vc = rda_conv(model, layer, "v", &xv, din)?;
    emit(&format!("rda_q_conv-{layer}"), &qc);
    emit(&format!("rda_k_conv-{layer}"), &kc);
    emit(&format!("rda_v_conv-{layer}"), &vc);

    // g1 = lower_bound * sigmoid(-(A * (f_b(f_a(cur)) + dt_b))), A per head.
    let tsr = spec.time_step_rank as usize;
    let fa = fork_mm(
        model,
        &format!("blk.{layer}.ssm_f_a.weight"),
        cur,
        h,
        tsr,
        nt,
    )?;
    let mut g1 = fork_mm(
        model,
        &format!("blk.{layer}.ssm_f_b.weight"),
        &fa,
        tsr,
        din,
        nt,
    )?;
    let dt_b = w("ssm_dt.bias")?;
    let ssm_a = w("ssm_a")?;
    for t in 0..nt {
        for c in 0..din {
            let idx = c + t * din;
            let raw = (g1[idx] + dt_b[c]) * ssm_a[c / hd];
            g1[idx] = hy.gate_lower_bound * sigmoid(-raw);
        }
    }

    // Erase/write gates fold into V; the scan's beta is sigmoid(beta).
    let beta = fork_mm(
        model,
        &format!("blk.{layer}.ssm_beta.weight"),
        cur,
        h,
        heads,
        nt,
    )?;
    let delta = fork_mm(
        model,
        &format!("blk.{layer}.ssm_rda.weight"),
        cur,
        h,
        heads,
        nt,
    )?;
    let mut vs = vc.clone();
    for t in 0..nt {
        for head in 0..heads {
            let bi = beta[head + t * heads];
            let de = bi + delta[head + t * heads];
            let fold = (softplus(-bi) - softplus(-de)).exp();
            for d in 0..hd {
                vs[d + head * hd + t * din] *= fold;
            }
        }
    }
    emit(&format!("rda_v_scaled-{layer}"), &vs);
    emit(&format!("rda_g1-{layer}"), &g1);
    let bscan: Vec<f32> = beta.iter().map(|&b| sigmoid(b)).collect();
    emit(&format!("rda_beta-{layer}"), &bscan);

    l2_norm_cols(&mut qc, hd, 1e-6);
    l2_norm_cols(&mut kc, hd, 1e-6);
    // The 1e-6 l2 eps is the fork's hard-coded value (raon1.cpp's F.normalize
    // eps), not a checkpoint field; nothing here reads it from the header.

    // Sequential delta-rule scan (the fork chunks at 16, same law):
    // per-channel decay on the k-index rows, delta write k⊗δ, then the
    // read q^T·S after the write — exactly the fork's GDN kernel order.
    // The state matrix s is f32 on purpose: it matches the fork's state
    // dtype, so scan divergence measured here is what the kernel port
    // would see too.
    let qs = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; din * nt];
    for head in 0..heads {
        // s[u + w*hd] = S[u][w]: u is the k-dim index, w the v-dim index.
        let mut s = vec![0.0f32; hd * hd];
        let mut dvec = vec![0.0f32; hd];
        for t in 0..nt {
            for u in 0..hd {
                let decay = g1[u + head * hd + t * din].exp();
                for w in 0..hd {
                    s[u + w * hd] *= decay;
                }
            }
            for w in 0..hd {
                let mut sk = 0.0f64;
                for u in 0..hd {
                    sk += s[u + w * hd] as f64 * kc[u + head * hd + t * din] as f64;
                }
                dvec[w] = ((vs[w + head * hd + t * din] as f64 - sk)
                    * bscan[head + t * heads] as f64) as f32;
            }
            for u in 0..hd {
                let ku = kc[u + head * hd + t * din];
                for w in 0..hd {
                    s[u + w * hd] += ku * dvec[w];
                }
            }
            for w in 0..hd {
                let mut acc = 0.0f64;
                for u in 0..hd {
                    acc += s[u + w * hd] as f64 * qc[u + head * hd + t * din] as f64;
                }
                out[w + head * hd + t * din] = (acc * qs as f64) as f32;
            }
        }
    }
    emit(&format!("rda_scan_out-{layer}"), &out);

    let o_norm = rms_norm_scaled(&out, &w("ssm_norm.weight")?, hd, hy.norm_eps);
    emit(&format!("rda_normed-{layer}"), &o_norm);
    let g2 = fork_mm(model, &format!("blk.{layer}.ssm_g.weight"), cur, h, din, nt)?;
    emit(&format!("rda_g2-{layer}"), &g2);
    let mut gated = vec![0.0f32; din * nt];
    for (slot, (&o, &g)) in gated.iter_mut().zip(o_norm.iter().zip(&g2)) {
        *slot = o * sigmoid(g);
    }
    let out_mm = fork_mm(
        model,
        &format!("blk.{layer}.attn_output.weight"),
        &gated,
        din,
        h,
        nt,
    )?;
    emit(&format!("rda_out-{layer}"), &out_mm);
    Ok(out_mm)
}

/// The absorbed MLA layer: NoPE concat, latent attention, output gate.
fn mla_layer(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    cur: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let heads = spec.heads as usize;
    let nt = cur.len() / h;
    let rank = spec.kv_latent as usize;
    let nr = hy.rope_dim;
    let nk = spec.key_mla as usize;
    let nn = nk - nr;
    let qw = spec.q_lora as usize;
    let vh = spec.v_head as usize;

    let w = |name: &str| model.read_f32(&format!("blk.{layer}.{name}"));

    // Query: down-project, norm, up-project; per head [nope | rope-extra].
    let qa = fork_mm(
        model,
        &format!("blk.{layer}.attn_q_a.weight"),
        cur,
        h,
        qw,
        nt,
    )?;
    let qan = rms_norm_scaled(&qa, &w("attn_q_a_norm.weight")?, qw, hy.latent_eps);
    let q = fork_mm(
        model,
        &format!("blk.{layer}.attn_q_b.weight"),
        &qan,
        qw,
        heads * nk,
        nt,
    )?;

    // KV: normed latent + raw NoPE extra, one shared KV head.
    let kv = fork_mm(
        model,
        &format!("blk.{layer}.attn_kv_a_mqa.weight"),
        cur,
        h,
        rank + nr,
        nt,
    )?;
    let mut latent = vec![0.0f32; rank * nt];
    for t in 0..nt {
        latent[t * rank..(t + 1) * rank]
            .copy_from_slice(&kv[t * (rank + nr)..t * (rank + nr) + rank]);
    }
    let latent = rms_norm_scaled(&latent, &w("attn_kv_a_norm.weight")?, rank, hy.latent_eps);

    let wk_b = w("attn_k_b.weight")?;
    let wv_b = w("attn_v_b.weight")?;
    // 1/sqrt(nk) is the fork's hard-coded query scale (raon1.cpp), not a
    // checkpoint field.
    let scale = 1.0 / (nk as f32).sqrt();

    let mut out = vec![0.0f32; heads * vh * nt];
    for head in 0..heads {
        for t in 0..nt {
            // Absorbed key: q_nope through the per-head wk_b [nn, rank].
            let mut qk = vec![0.0f32; rank];
            for c in 0..rank {
                let mut acc = 0.0f64;
                for d in 0..nn {
                    acc += wk_b[d + c * nn + head * nn * rank] as f64
                        * q[d + head * nk + t * heads * nk] as f64;
                }
                qk[c] = acc as f32;
            }
            // Causal softmax over the shared key; the denominator and the
            // latent mix accumulate in f64 like the dot products above (the
            // fork accumulates f32, which is fine at 42 tokens but not a
            // whole-context oracle).
            let mut probs = vec![0.0f32; t + 1];
            let mut maxv = f32::NEG_INFINITY;
            for tk in 0..=t {
                let mut acc = 0.0f64;
                for d in 0..rank {
                    acc += latent[d + tk * rank] as f64 * qk[d] as f64;
                }
                for d in 0..nr {
                    acc += kv[rank + d + tk * (rank + nr)] as f64
                        * q[nn + d + head * nk + t * heads * nk] as f64;
                }
                probs[tk] = (acc as f32) * scale;
                maxv = maxv.max(probs[tk]);
            }
            let mut sum = 0.0f64;
            for p in probs.iter_mut() {
                *p = (*p - maxv).exp();
                sum += *p as f64;
            }
            // Absorbed order: mix the latents, then project through wv_b.
            let mut mix = vec![0.0f64; rank];
            for tk in 0..=t {
                let p = probs[tk] as f64 / sum;
                for c in 0..rank {
                    mix[c] += p * latent[c + tk * rank] as f64;
                }
            }
            for v in 0..vh {
                let mut acc = 0.0f64;
                for c in 0..rank {
                    acc += wv_b[c + v * rank + head * rank * vh] as f64 * mix[c];
                }
                out[v + head * vh + t * heads * vh] = acc as f32;
            }
        }
    }

    let gate = fork_mm(
        model,
        &format!("blk.{layer}.attn_gate.weight"),
        cur,
        h,
        heads * vh,
        nt,
    )?;
    for (slot, &g) in out.iter_mut().zip(gate.iter()) {
        *slot *= sigmoid(g);
    }
    emit(&format!("mla_gated-{layer}"), &out);
    fork_mm(
        model,
        &format!("blk.{layer}.attn_output.weight"),
        &out,
        heads * vh,
        h,
        nt,
    )
}

/// The shared-expert FFN: SiTU with parallel gate/up over the MoE width.
fn shexp_ffn(
    model: &Raon1Model,
    cur: &[f32],
    layer: usize,
    spec: Raon1Spec,
    hy: &Hyper,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let mi = spec.moe_inter as usize;
    let nt = cur.len() / h;
    let gate = fork_mm(
        model,
        &format!("blk.{layer}.ffn_gate_shexp.weight"),
        cur,
        h,
        mi,
        nt,
    )?;
    let up = fork_mm(
        model,
        &format!("blk.{layer}.ffn_up_shexp.weight"),
        cur,
        h,
        mi,
        nt,
    )?;
    emit(&format!("ffn_gate-{layer}"), &gate);
    emit(&format!("ffn_up-{layer}"), &up);
    let s = situ(&gate, &up, hy.situ_beta, hy.situ_linear_beta);
    emit(&format!("ffn_situ-{layer}"), &s);
    fork_mm(
        model,
        &format!("blk.{layer}.ffn_down_shexp.weight"),
        &s,
        mi,
        h,
        nt,
    )
}

/// The MoE block: sigmoid routing with a selection bias, top-k over the
/// biased scores, unbiased weights normalized by their clamped sum.
/// `forced` (used only by the fixture test) replaces the biased top-k pick
/// with a known selection so the expert columns compare positionally.
#[allow(clippy::too_many_arguments)]
fn moe_layer(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    cur: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
    forced: Option<&[u8]>,
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let nt = cur.len() / h;
    let ne = spec.experts as usize;
    let nu = hy.used_experts;
    let mi = spec.moe_inter as usize;

    let logits = fork_mm(
        model,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        cur,
        h,
        ne,
        nt,
    )?;
    let probs: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
    emit(&format!("ffn_moe_logits-{layer}"), &logits);
    emit(&format!("ffn_moe_probs-{layer}"), &probs);

    let bias = model.read_f32(&format!("blk.{layer}.exp_probs_b.bias"))?;
    let mut sel = vec![0.0f32; ne * nt];
    for t in 0..nt {
        for e in 0..ne {
            sel[e + t * ne] = probs[e + t * ne] + bias[e];
        }
    }
    emit(&format!("ffn_moe_probs_biased-{layer}"), &sel);

    // Top-k per token by biased score (descending, ties to the lower index).
    // total_cmp: the biased scores can be exactly equal or (on a bad file)
    // NaN, and a NaN-poisoned partial_cmp would silently drop candidates.
    let mut top = vec![(0usize, 0.0f32); nu * nt];
    for t in 0..nt {
        let forced_t = forced.map(|f| &f[t * nu..(t + 1) * nu]);
        if let Some(ids) = forced_t {
            for (slot, &e) in ids.iter().enumerate() {
                top[slot + t * nu] = (e as usize, probs[e as usize + t * ne]);
            }
            continue;
        }
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| sel[b + t * ne].total_cmp(&sel[a + t * ne]));
        for (slot, &e) in idx.iter().enumerate().take(nu) {
            top[slot + t * nu] = (e, probs[e + t * ne]);
        }
    }
    let weights: Vec<f32> = (0..nu * nt).map(|i| top[i].1).collect();
    emit(&format!("ffn_moe_weights-{layer}"), &weights);

    let mut wsum = vec![0.0f32; nt];
    for t in 0..nt {
        wsum[t] = (0..nu).map(|u| weights[u + t * nu]).sum();
    }
    emit(&format!("ffn_moe_weights_sum-{layer}"), &wsum);
    let clamped: Vec<f32> = wsum.iter().map(|&v| v.max(WEIGHT_SUM_MIN)).collect();
    emit(&format!("ffn_moe_weights_sum_clamped-{layer}"), &clamped);
    let mut norm = vec![0.0f32; nu * nt];
    for t in 0..nt {
        for u in 0..nu {
            norm[u + t * nu] = if hy.weights_norm {
                weights[u + t * nu] / clamped[t]
            } else {
                weights[u + t * nu]
            };
        }
    }
    emit(&format!("ffn_moe_weights_norm-{layer}"), &norm);

    // Per selected expert and token: gate/up/situ/down, slot layout
    // [moe_inter, used, nt] with slot order = the top-k order.
    let (up_w, up_ty) = model.read_f32_typed(&format!("blk.{layer}.ffn_up_exps.weight"))?;
    let (gate_w, gate_ty) = model.read_f32_typed(&format!("blk.{layer}.ffn_gate_exps.weight"))?;
    let (down_w, down_ty) = model.read_f32_typed(&format!("blk.{layer}.ffn_down_exps.weight"))?;
    let span = h * mi;
    let mut moe_out = vec![0.0f32; h * nt];
    let mut gbuf = vec![0.0f32; mi * nu * nt];
    let mut ubuf = vec![0.0f32; mi * nu * nt];
    let mut sbuf = vec![0.0f32; mi * nu * nt];
    let mut dbuf = vec![0.0f32; h * nu * nt];
    let mut wbuf = vec![0.0f32; h * nu * nt];
    for t in 0..nt {
        let xt = &cur[t * h..(t + 1) * h];
        for u in 0..nu {
            let e = top[u + t * nu].0;
            let g = fork_mv(&gate_w[e * span..][..span], gate_ty, xt, h, mi);
            let up = fork_mv(&up_w[e * span..][..span], up_ty, xt, h, mi);
            let s = situ(&g, &up, hy.situ_beta, hy.situ_linear_beta);
            let dn = fork_mv(&down_w[e * mi * h..][..mi * h], down_ty, &s, mi, h);
            for f in 0..mi {
                gbuf[f + u * mi + t * mi * nu] = g[f];
                ubuf[f + u * mi + t * mi * nu] = up[f];
                sbuf[f + u * mi + t * mi * nu] = s[f];
            }
            let wv = norm[u + t * nu];
            for k in 0..h {
                dbuf[k + u * h + t * h * nu] = dn[k];
                wbuf[k + u * h + t * h * nu] = dn[k] * wv;
                moe_out[k + t * h] += dn[k] * wv;
            }
        }
    }
    emit(&format!("ffn_moe_gate-{layer}"), &gbuf);
    emit(&format!("ffn_moe_up-{layer}"), &ubuf);
    emit(&format!("ffn_moe_situ-{layer}"), &sbuf);
    emit(&format!("ffn_moe_down-{layer}"), &dbuf);
    emit(&format!("ffn_moe_weighted-{layer}"), &wbuf);
    emit(&format!("ffn_moe_out-{layer}"), &moe_out);

    let shared = shexp_ffn(model, cur, layer, spec, hy, emit)?;
    let mut out = moe_out;
    for (slot, &sv) in out.iter_mut().zip(&shared) {
        *slot += sv;
    }
    Ok(out)
}

/// The dense FFN of the leading block (wider than the MoE experts).
fn dense_ffn(
    model: &Raon1Model,
    cur: &[f32],
    layer: usize,
    spec: Raon1Spec,
    hy: &Hyper,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let di = spec.dense_inter as usize;
    let nt = cur.len() / h;
    let gate = fork_mm(
        model,
        &format!("blk.{layer}.ffn_gate.weight"),
        cur,
        h,
        di,
        nt,
    )?;
    let up = fork_mm(model, &format!("blk.{layer}.ffn_up.weight"), cur, h, di, nt)?;
    emit(&format!("ffn_gate-{layer}"), &gate);
    emit(&format!("ffn_up-{layer}"), &up);
    let s = situ(&gate, &up, hy.situ_beta, hy.situ_linear_beta);
    emit(&format!("ffn_situ-{layer}"), &s);
    fork_mm(
        model,
        &format!("blk.{layer}.ffn_down.weight"),
        &s,
        di,
        h,
        nt,
    )
}

/// One layer's attention half: mHC site, norm, RDA-or-MLA mixer, mHC fold.
/// Returns the half's output residual (`hc_attn_out-{layer}` layout).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_half(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    inp: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let attn = hc_site(model, spec, hy, inp, layer, "attn", emit)?;
    let norm_w = model.read_f32(&format!("blk.{layer}.attn_norm.weight"))?;
    let mut cur = rms_norm_scaled(&attn.branch, &norm_w, h, hy.norm_eps);
    emit(&format!("attn_norm-{layer}"), &cur);
    cur = if spec.mla_layers.contains(&layer) {
        mla_layer(model, spec, hy, &cur, layer, emit)?
    } else {
        rda_layer(model, spec, hy, &cur, layer, emit)?
    };
    emit(&format!("hc_post-{layer}"), &attn.post);
    emit(&format!("hc_comb-{layer}"), &attn.comb);
    let out = hc_post_apply(inp, &cur, &attn, hy, spec);
    emit(&format!("hc_attn_out-{layer}"), &out);
    Ok(out)
}

/// One layer's FFN half: mHC site, norm, dense-or-MoE FFN, mHC fold. `forced`
/// is the fixture test's teacher-forcing hook into the MoE top-k.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_half(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    inp: &[f32],
    layer: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
    forced: Option<&[u8]>,
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let ffn = hc_site(model, spec, hy, inp, layer, "ffn", emit)?;
    let norm_w = model.read_f32(&format!("blk.{layer}.ffn_norm.weight"))?;
    let mut cur = rms_norm_scaled(&ffn.branch, &norm_w, h, hy.norm_eps);
    emit(&format!("ffn_norm-{layer}"), &cur);
    cur = if layer < spec.leading_dense {
        dense_ffn(model, &cur, layer, spec, hy, emit)?
    } else {
        moe_layer(model, spec, hy, &cur, layer, emit, forced)?
    };
    emit(&format!("hc_post-{layer}"), &ffn.post);
    emit(&format!("hc_comb-{layer}"), &ffn.comb);
    let out = hc_post_apply(inp, &cur, &ffn, hy, spec);
    emit(&format!("l_out-{layer}"), &out);
    Ok(out)
}

/// The output head: mean over streams, last token, norm, untied matmul.
/// `inp` is the final `l_out` residual (`[hidden, streams, nt]`).
pub(crate) fn head(
    model: &Raon1Model,
    spec: Raon1Spec,
    hy: &Hyper,
    inp: &[f32],
    nt: usize,
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, EngError> {
    let h = spec.hidden as usize;
    let hc = hy.streams;
    let mut mean = vec![0.0f32; h];
    for s in 0..hc {
        for k in 0..h {
            mean[k] += inp[k + s * h + (nt - 1) * h * hc];
        }
    }
    for v in mean.iter_mut() {
        *v /= hc as f32;
    }
    let normed = rms_norm_scaled(
        &mean,
        &model.read_f32("output_norm.weight")?,
        h,
        hy.norm_eps,
    );
    emit("result_norm", &normed);
    let logits = fork_mm(model, "output.weight", &normed, h, spec.vocab as usize, 1)?;
    emit("result_output", &logits);
    Ok(logits)
}

/// The full forward pass, emitting every named graph stage in fork graph
/// order. `emit(name, data)` receives ggml-layout F32 contents.
pub fn forward(
    model: &Raon1Model,
    tokens: &[u32],
    emit: &mut dyn FnMut(&str, &[f32]),
) -> Result<(), EngError> {
    let spec = model.spec();
    if tokens.is_empty() {
        return Err(reject("a forward pass needs at least one token"));
    }
    if let Some(&tok) = tokens.iter().find(|&&t| t >= spec.vocab as u32) {
        return Err(reject(format!(
            "token id {tok} is at or above the vocab size {}",
            spec.vocab
        )));
    }
    let hy = hyper(model, spec)?;
    let h = spec.hidden as usize;
    let hc = hy.streams;
    let nt = tokens.len();

    // Embedding, repeated into all hc streams.
    let embd_w = model.read_f32("token_embd.weight")?;
    let mut embd = vec![0.0f32; h * nt];
    for (t, &tok) in tokens.iter().enumerate() {
        embd[t * h..(t + 1) * h].copy_from_slice(&embd_w[tok as usize * h..(tok as usize + 1) * h]);
    }
    emit("embd", &embd);
    let mut inp = vec![0.0f32; h * hc * nt];
    for t in 0..nt {
        for s in 0..hc {
            for k in 0..h {
                inp[k + s * h + t * h * hc] = embd[k + t * h];
            }
        }
    }

    for layer in 0..spec.layers {
        inp = attn_half(model, spec, &hy, &inp, layer, emit)?;
        inp = ffn_half(model, spec, &hy, &inp, layer, emit, None)?;
    }
    head(model, spec, &hy, &inp, nt, emit)?;
    Ok(())
}
