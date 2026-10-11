//! Flash-Next DeltaNet decode step (design section "DeltaNet").

use super::math::{expect_len, sigmoid, silu, softplus};
use crate::EngError;
use crate::family::project::Projector;

/// Checked DeltaNet dimensions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeltaShape {
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
    conv_kernel: usize,
    hidden: usize,
    eps: f32,
    conv_dim: usize,
    state_len: usize,
    ring_len: usize,
}

impl DeltaShape {
    /// Checks the head counts, widths, kernel, hidden width, and epsilon.
    pub(crate) fn new(
        key_heads: u32,
        key_dim: u32,
        value_heads: u32,
        value_dim: u32,
        conv_kernel: u32,
        hidden: u32,
        eps: f32,
    ) -> Result<Self, EngError> {
        if key_heads == 0 || key_dim == 0 || value_heads == 0 || value_dim == 0 || hidden == 0 {
            return Err(EngError::Checkpoint(
                "delta dimensions must exceed zero".into(),
            ));
        }
        if conv_kernel < 2 {
            return Err(EngError::Checkpoint(
                "delta convolution kernel must be at least 2".into(),
            ));
        }
        if !value_heads.is_multiple_of(key_heads) {
            return Err(EngError::Checkpoint(
                "delta value heads must divide by key heads".into(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(EngError::Checkpoint("delta epsilon is invalid".into()));
        }
        let kh = key_heads as usize;
        let kd = key_dim as usize;
        let vh = value_heads as usize;
        let vd = value_dim as usize;
        let ck = conv_kernel as usize;
        let (conv_dim, state_len, ring_len) = checked_sizes(kh, kd, vh, vd, ck)
            .ok_or_else(|| EngError::Checkpoint("delta dimensions overflow".into()))?;
        Ok(Self {
            key_heads: kh,
            key_dim: kd,
            value_heads: vh,
            value_dim: vd,
            conv_kernel: ck,
            hidden: hidden as usize,
            eps,
            conv_dim,
            state_len,
            ring_len,
        })
    }

    /// Key heads.
    pub(crate) fn key_heads(&self) -> usize {
        self.key_heads
    }
    /// Width of one key head.
    pub(crate) fn key_dim(&self) -> usize {
        self.key_dim
    }
    /// Value heads.
    pub(crate) fn value_heads(&self) -> usize {
        self.value_heads
    }
    /// Width of one value head.
    pub(crate) fn value_dim(&self) -> usize {
        self.value_dim
    }
    /// Convolution kernel length.
    pub(crate) fn conv_kernel(&self) -> usize {
        self.conv_kernel
    }
    /// Convolution channels `2 * key_heads * key_dim + value_heads * value_dim`.
    pub(crate) fn conv_dim(&self) -> usize {
        self.conv_dim
    }
    /// RMS normalization epsilon.
    pub(crate) fn eps(&self) -> f32 {
        self.eps
    }
}

/// Returns `(conv_dim, state_len, ring_len)`, or `None` when a size overflows `usize`.
fn checked_sizes(
    kh: usize,
    kd: usize,
    vh: usize,
    vd: usize,
    ck: usize,
) -> Option<(usize, usize, usize)> {
    let key_width = kh.checked_mul(kd)?;
    let conv_dim = key_width.checked_mul(2)?.checked_add(vh.checked_mul(vd)?)?;
    let state_len = vh.checked_mul(kd)?.checked_mul(vd)?;
    let ring_len = conv_dim.checked_mul(ck.checked_sub(1)?)?;
    Some((conv_dim, state_len, ring_len))
}

/// Weights of one DeltaNet layer.
pub(crate) struct DeltaWeights<'a, W> {
    /// Fused query, key, and value projection `[conv_dim, hidden]`.
    pub(crate) qkv: &'a W,
    /// Output gate projection `[value_heads * value_dim, hidden]`.
    pub(crate) z: &'a W,
    /// Beta projection `[value_heads, hidden]`.
    pub(crate) b: &'a W,
    /// Alpha projection `[value_heads, hidden]`.
    pub(crate) a: &'a W,
    /// Output projection `[hidden, value_heads * value_dim]`.
    pub(crate) out: &'a W,
    /// Convolution weights, length `conv_dim * conv_kernel`, row `c` at `c * conv_kernel`.
    pub(crate) conv: &'a [f32],
    /// `A_log`, length `value_heads`.
    pub(crate) a_log: &'a [f32],
    /// `dt_bias`, length `value_heads`.
    pub(crate) dt_bias: &'a [f32],
    /// Output norm weight, length `value_dim`, shared by all value heads.
    pub(crate) norm: &'a [f32],
}

/// Recurrent state and convolution ring of one DeltaNet layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeltaState {
    rec: Vec<f32>,
    ring: Vec<f32>,
}

impl DeltaState {
    /// Returns a zero state sized for `shape`.
    pub(crate) fn new(shape: &DeltaShape) -> Self {
        Self {
            rec: vec![0.0; shape.state_len],
            ring: vec![0.0; shape.ring_len],
        }
    }

    /// Host bytes held by the state.
    pub(crate) fn byte_len(&self) -> usize {
        (self.rec.len() + self.ring.len()) * 4
    }
}

#[cfg(test)]
impl DeltaState {
    /// Recurrent state `[value_heads, key_dim, value_dim]`, row-major (test observer).
    pub(crate) fn rec(&self) -> &[f32] {
        &self.rec
    }

    /// Convolution ring `[conv_dim, conv_kernel - 1]`, row-major (test observer).
    pub(crate) fn ring(&self) -> &[f32] {
        &self.ring
    }
}

/// Runs one decode step and returns the layer output of length `hidden`.
pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &DeltaShape,
    weights: &DeltaWeights<'_, P::Weight>,
    state: &mut DeltaState,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let kh = shape.key_heads;
    let kd = shape.key_dim;
    let vh = shape.value_heads;
    let vd = shape.value_dim;
    let ck = shape.conv_kernel;
    let eps = shape.eps;
    let taps = ck - 1;
    let kw = kh * kd;
    let vw = vh * vd;
    let conv_len = shape
        .conv_dim
        .checked_mul(ck)
        .ok_or_else(|| EngError::Checkpoint("delta dimensions overflow".into()))?;

    expect_len(x, shape.hidden, "delta input")?;
    expect_len(weights.conv, conv_len, "delta conv")?;
    expect_len(weights.a_log, vh, "delta A_log")?;
    expect_len(weights.dt_bias, vh, "delta dt_bias")?;
    expect_len(weights.norm, vd, "delta norm")?;
    expect_len(&state.rec, shape.state_len, "delta state")?;
    expect_len(&state.ring, shape.ring_len, "delta ring")?;

    let u = projector.linear(weights.qkv, x)?;
    expect_len(&u, shape.conv_dim, "delta qkv")?;
    let z = projector.linear(weights.z, x)?;
    expect_len(&z, vw, "delta z")?;
    let bb = projector.linear(weights.b, x)?;
    expect_len(&bb, vh, "delta b")?;
    let aa = projector.linear(weights.a, x)?;
    expect_len(&aa, vh, "delta a")?;

    let mut y = Vec::with_capacity(shape.conv_dim);
    let mut new_ring = Vec::with_capacity(shape.ring_len);
    for (c, &uc) in u.iter().enumerate() {
        let mut val = weights.conv[c * ck + taps] * uc;
        for t in 0..taps {
            val += weights.conv[c * ck + t] * state.ring[c * taps + t];
        }
        y.push(silu(val));
        for t in 0..taps - 1 {
            new_ring.push(state.ring[c * taps + t + 1]);
        }
        new_ring.push(uc);
    }

    let mut q = Vec::with_capacity(kw);
    let mut k = Vec::with_capacity(kw);
    for g in 0..kh {
        let query = &y[g * kd..(g + 1) * kd];
        let query_den =
            (1e-6 + query.iter().map(|v| v * v).sum::<f32>()).sqrt() * (kd as f32).sqrt();
        q.extend(query.iter().map(|v| v / query_den));
        let key = &y[kw + g * kd..kw + (g + 1) * kd];
        let key_den = (1e-6 + key.iter().map(|v| v * v).sum::<f32>()).sqrt();
        k.extend(key.iter().map(|v| v / key_den));
    }

    let rep = vh / kh;
    let mut new_rec = state.rec.clone();
    let mut o = Vec::with_capacity(vw);
    for h in 0..vh {
        let g = h / rep;
        let alpha = (-(weights.a_log[h].exp()) * softplus(aa[h] + weights.dt_bias[h])).exp();
        let beta = sigmoid(bb[h]);
        let s = &mut new_rec[h * kd * vd..(h + 1) * kd * vd];
        for v in s.iter_mut() {
            *v *= alpha;
        }
        let mut delta = Vec::with_capacity(vd);
        for j in 0..vd {
            let mut prev = 0.0f32;
            for i in 0..kd {
                prev += k[g * kd + i] * s[i * vd + j];
            }
            delta.push((y[2 * kw + h * vd + j] - prev) * beta);
        }
        for i in 0..kd {
            for (j, &dj) in delta.iter().enumerate() {
                s[i * vd + j] += k[g * kd + i] * dj;
            }
        }
        let mut core = Vec::with_capacity(vd);
        for j in 0..vd {
            let mut acc = 0.0f32;
            for i in 0..kd {
                acc += q[g * kd + i] * s[i * vd + j];
            }
            core.push(acc);
        }
        let ms = core.iter().map(|c| c * c).sum::<f32>() / vd as f32;
        for (j, &cj) in core.iter().enumerate() {
            o.push(cj * sigmoid(z[h * vd + j]) * weights.norm[j] / (ms + eps).sqrt());
        }
    }

    let result = projector.linear(weights.out, &o)?;
    expect_len(&result, shape.hidden, "delta output")?;
    state.ring = new_ring;
    state.rec = new_rec;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};

    /// QKV rows of the fixture. Input `[1, -1]` yields these values.
    const QKV_VALUES: [f32; 16] = [
        0.5, 1.0, -0.3, 0.8, 0.2, -0.6, 0.9, 0.4, -0.7, 0.3, 0.6, -0.2, 0.1, 0.5, -0.4, 0.7,
    ];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn assert_all_close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(close(*a, *e), "{what} element {i}: {a} vs {e}");
        }
    }

    /// Builds a `[values.len(), 2]` matrix whose row r is `[v/2, -v/2]`, so `x = [1, -1]` yields `values`.
    fn split_rows(values: &[f32]) -> DenseWeight {
        DenseWeight::matrix(
            values.len(),
            2,
            values.iter().flat_map(|v| [v / 2.0, -v / 2.0]).collect(),
        )
    }

    struct Fixture {
        shape: DeltaShape,
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

    impl Fixture {
        fn new() -> Self {
            let conv: Vec<f32> = (0..16)
                .flat_map(|c| {
                    if c % 2 == 0 {
                        [0.25, 0.5, 1.0]
                    } else {
                        [-0.5, 0.75, 2.0]
                    }
                })
                .collect();
            Self {
                shape: DeltaShape::new(2, 2, 4, 2, 3, 2, 1e-6).unwrap(),
                qkv: split_rows(&QKV_VALUES),
                z: split_rows(&[0.3, -0.2, 1.0, 0.5, -0.6, 0.4, 0.2, -1.0]),
                b: split_rows(&[0.0, 1.0, -0.5, 0.25]),
                a: split_rows(&[0.5, -1.0, 0.0, 1.5]),
                out: DenseWeight::matrix(
                    2,
                    8,
                    vec![
                        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
                        -1.0,
                    ],
                ),
                conv,
                a_log: vec![0.0, -1.0, 0.5, -0.5],
                dt_bias: vec![0.1, 0.2, -0.3, 0.0],
                norm: vec![0.5, 2.0],
            }
        }

        fn weights(&self) -> DeltaWeights<'_, DenseWeight> {
            DeltaWeights {
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

    #[test]
    fn shape_rejects_invalid_dimensions() {
        assert!(matches!(
            DeltaShape::new(0, 2, 4, 2, 3, 2, 1e-6),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            DeltaShape::new(2, 2, 4, 2, 1, 2, 1e-6),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            DeltaShape::new(2, 2, 3, 2, 3, 2, 1e-6),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            DeltaShape::new(2, 2, 4, 2, 3, 2, f32::NAN),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn zero_state_has_expected_size() {
        let f = Fixture::new();
        let state = DeltaState::new(&f.shape);
        assert_eq!(state.rec.len(), 16);
        assert_eq!(state.ring.len(), 32);
        assert!(state.rec.iter().chain(&state.ring).all(|v| *v == 0.0));
        assert_eq!(state.byte_len(), 192);
    }

    #[test]
    fn two_steps_match_hand_values() {
        let f = Fixture::new();
        let weights = f.weights();
        let mut projector = DenseProjector::default();
        let mut state = DeltaState::new(&f.shape);

        let out1 = step(&mut projector, &f.shape, &weights, &mut state, &[1.0, -1.0]).unwrap();
        assert_all_close(&out1, &[-0.305_751_4, -0.830_060_5], "step 1 output");
        for (c, pair) in state.ring.as_chunks::<2>().0.iter().enumerate() {
            assert!(pair[0] == 0.0, "ring[{}] is {} vs 0.0", 2 * c, pair[0]);
            assert!(
                close(pair[1], QKV_VALUES[c]),
                "ring[{}] is {} vs {}",
                2 * c + 1,
                pair[1],
                QKV_VALUES[c]
            );
        }

        let out2 = step(&mut projector, &f.shape, &weights, &mut state, &[0.5, 2.0]).unwrap();
        assert_all_close(&out2, &[0.029_617_6, 2.438_123], "step 2 output");
        assert_all_close(
            &state.ring,
            &[
                0.5, -0.375, 1.0, -0.75, -0.3, 0.225, 0.8, -0.6, 0.2, -0.15, -0.6, 0.45, 0.9,
                -0.675, 0.4, -0.3, -0.7, 0.525, 0.3, -0.225, 0.6, -0.45, -0.2, 0.15, 0.1, -0.075,
                0.5, -0.375, -0.4, 0.3, 0.7, -0.525,
            ],
            "ring after step 2",
        );
        assert_all_close(
            &state.rec,
            &[
                -0.025_702_4,
                0.040_271_4,
                0.077_236_4,
                -0.099_545_9,
                0.062_325_2,
                -0.027_296,
                -0.132_549_1,
                0.071_498_2,
                0.007_678,
                0.099_928_4,
                0.007_313_4,
                0.092_908,
                -0.051_430_5,
                0.311_731_8,
                -0.041_797_3,
                0.230_969_8,
            ],
            "rec after step 2",
        );
    }

    #[test]
    fn wrong_input_leaves_state_unchanged() {
        let f = Fixture::new();
        let mut projector = DenseProjector::default();
        let mut state = DeltaState::new(&f.shape);
        let result = step(
            &mut projector,
            &f.shape,
            &f.weights(),
            &mut state,
            &[1.0, 2.0, 3.0],
        );
        assert!(matches!(result, Err(EngError::BadRequest(_))));
        assert_eq!(projector.calls, 0);
        assert_eq!(state, DeltaState::new(&f.shape));
    }

    #[test]
    fn failed_output_projection_leaves_state_unchanged() {
        let mut f = Fixture::new();
        f.out = DenseWeight::matrix(2, 7, vec![1.0; 14]);
        let mut projector = DenseProjector::default();
        let mut state = DeltaState::new(&f.shape);
        let result = step(
            &mut projector,
            &f.shape,
            &f.weights(),
            &mut state,
            &[1.0, -1.0],
        );
        assert!(matches!(result, Err(EngError::BadRequest(_))));
        assert_eq!(state, DeltaState::new(&f.shape));
    }

    #[test]
    fn norm_weight_must_have_value_dim_length() {
        let mut f = Fixture::new();
        f.norm = vec![1.0; 8];
        let mut projector = DenseProjector::default();
        let mut state = DeltaState::new(&f.shape);
        let result = step(
            &mut projector,
            &f.shape,
            &f.weights(),
            &mut state,
            &[1.0, -1.0],
        );
        assert!(matches!(result, Err(EngError::BadRequest(_))));
    }
}
