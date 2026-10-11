//! Flash-Next hyper-connection read and apply (design section "Hyper-connection").

use super::math::{expect_len, rms0, sigmoid, silu};
use crate::EngError;
use crate::family::project::Projector;

/// Checked hyper-connection dimensions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HyperShape {
    branches: usize,
    hidden: usize,
    rank: usize,
    eps: f32,
}

impl HyperShape {
    /// Checks the branch count, hidden width, low rank, and epsilon.
    pub(crate) fn new(branches: u32, hidden: u32, rank: u32, eps: f32) -> Result<Self, EngError> {
        if branches == 0 || hidden == 0 || rank == 0 {
            return Err(EngError::Checkpoint(
                "hyper-connection dimensions must exceed zero".into(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(EngError::Checkpoint(
                "hyper-connection epsilon is invalid".into(),
            ));
        }
        let branches = branches as usize;
        let hidden = hidden as usize;
        branches
            .checked_mul(hidden)
            .ok_or_else(|| EngError::Checkpoint("hyper-connection width overflows".into()))?;
        Ok(Self {
            branches,
            hidden,
            rank: rank as usize,
            eps,
        })
    }

    /// Width `branches * hidden` of the residual stream.
    pub(crate) fn width(&self) -> usize {
        self.branches * self.hidden
    }

    /// Number of residual branches.
    pub(crate) fn branches(&self) -> usize {
        self.branches
    }

    /// Hidden width of one branch.
    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    /// Low rank of the mixing projections.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// RMS normalization epsilon.
    pub(crate) fn eps(&self) -> f32 {
        self.eps
    }
}

/// Weights of one hyper-connection.
pub(crate) struct HyperWeights<'a, W> {
    /// Branch norm weights, length `width`.
    pub(crate) norm: &'a [f32],
    /// Low-rank down projection `[rank, width]`.
    pub(crate) down: &'a W,
    /// Up projection `[width, rank]`.
    pub(crate) up: &'a W,
    /// Optional inject projection `[branches, width]`.
    pub(crate) inject: Option<&'a W>,
}

/// Result of a hyper-connection read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HyperRead {
    /// Mixed block input, length `hidden`.
    pub(crate) mixed: Vec<f32>,
    /// Per-branch apply gate, length `branches`, when the weights have an inject projection.
    pub(crate) gate: Option<Vec<f32>>,
}

/// Normalizes the branches, mixes them, and computes the optional apply gate.
pub(crate) fn read<P: Projector>(
    projector: &mut P,
    shape: &HyperShape,
    weights: &HyperWeights<'_, P::Weight>,
    hyper: &[f32],
) -> Result<HyperRead, EngError> {
    let width = shape.width();
    let branches = shape.branches();
    let hidden = shape.hidden();
    expect_len(hyper, width, "hyper residual")?;
    expect_len(weights.norm, width, "hyper norm")?;

    let mut n: Vec<f32> = Vec::with_capacity(width);
    for b in 0..branches {
        let range = b * hidden..(b + 1) * hidden;
        n.extend(rms0(
            &hyper[range.clone()],
            &weights.norm[range],
            shape.eps,
        )?);
    }

    let mut low = projector.linear(weights.down, &n)?;
    expect_len(&low, shape.rank, "hyper down output")?;
    for v in &mut low {
        *v = silu(*v / branches as f32);
    }

    let mix = projector.linear(weights.up, &low)?;
    expect_len(&mix, width, "hyper up output")?;
    let mut mixed = Vec::with_capacity(hidden);
    for d in 0..hidden {
        let mut sum = 0.0f32;
        for b in 0..branches {
            sum += sigmoid(mix[b * hidden + d]) * n[b * hidden + d];
        }
        mixed.push(sum / branches as f32);
    }

    let gate = match weights.inject {
        Some(inject) => {
            let inj = projector.linear(inject, &n)?;
            expect_len(&inj, branches, "hyper inject output")?;
            let mut gate = vec![0.0f32; branches];
            for b in 0..branches {
                gate[b] = 2.0 * sigmoid(inj[b] / branches as f32);
            }
            Some(gate)
        }
        None => None,
    };

    Ok(HyperRead { mixed, gate })
}

/// Adds `gate[b] * block[d]` to branch `b` of the residual.
pub(crate) fn apply(
    shape: &HyperShape,
    hyper: &mut [f32],
    gate: &[f32],
    block: &[f32],
) -> Result<(), EngError> {
    let branches = shape.branches();
    let hidden = shape.hidden();
    expect_len(hyper, shape.width(), "hyper residual")?;
    expect_len(gate, branches, "hyper gate")?;
    expect_len(block, hidden, "hyper block")?;
    for b in 0..branches {
        for d in 0..hidden {
            hyper[b * hidden + d] += gate[b] * block[d];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn shape() -> HyperShape {
        HyperShape::new(2, 2, 1, 0.0).unwrap()
    }

    #[test]
    fn shape_rejects_invalid_dimensions() {
        assert!(matches!(
            HyperShape::new(0, 2, 1, 0.0),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            HyperShape::new(2, 0, 1, 0.0),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            HyperShape::new(2, 2, 0, 0.0),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            HyperShape::new(2, 2, 1, -1.0),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            HyperShape::new(2, 2, 1, f32::NAN),
            Err(EngError::Checkpoint(_))
        ));
        let ok = HyperShape::new(2, 2, 1, 0.0).unwrap();
        assert_eq!(ok.width(), 4);
    }

    #[test]
    fn read_matches_hand_values() {
        let hyper = [3.0f32, 4.0, 1.0, 1.0];
        let norm = [0.0f32, 1.0, 0.0, 0.0];
        let down = DenseWeight::matrix(1, 4, vec![1.0, 0.5, -1.0, 2.0]);
        let up = DenseWeight::matrix(4, 1, vec![1.0, -1.0, 0.5, 2.0]);
        let inject = DenseWeight::matrix(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]);
        let weights = HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let mut p = DenseProjector::default();
        let out = read(&mut p, &shape(), &weights, &hyper).unwrap();
        assert_eq!(out.mixed.len(), 2);
        for (a, e) in out.mixed.iter().zip([0.650_989_4f32, 0.718_307_5]) {
            assert!(close(*a, e), "mixed {a} vs {e}");
        }
        let gate = out.gate.clone().unwrap();
        assert_eq!(gate.len(), 2);
        for (a, e) in gate.iter().zip([1.209_006_3f32, 1.0]) {
            assert!(close(*a, e), "gate {a} vs {e}");
        }
        assert_eq!(p.calls, 3);
    }

    #[test]
    fn read_uses_each_branch_norm_row() {
        let hyper = [3.0f32, 4.0, 1.0, 1.0];
        let norm = [0.0f32, 1.0, 0.0, 0.0];
        let down = DenseWeight::matrix(1, 4, vec![1.0, 0.5, -1.0, 2.0]);
        let up = DenseWeight::matrix(4, 1, vec![0.0; 4]);
        let weights = HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: None,
        };
        let mut p = DenseProjector::default();
        let out = read(&mut p, &shape(), &weights, &hyper).unwrap();
        assert_eq!(out.mixed.len(), 2);
        for (a, e) in out.mixed.iter().zip([0.462_132_f32, 0.815_685_4]) {
            assert!(close(*a, e), "mixed {a} vs {e}");
        }
        assert_eq!(out.gate, None);
        assert_eq!(p.calls, 2);
    }

    #[test]
    fn apply_adds_gate_times_block() {
        let mut hyper = [3.0f32, 4.0, 1.0, 1.0];
        let gate = [1.209_006_3f32, 1.0];
        let block = [10.0f32, -2.0];
        apply(&shape(), &mut hyper, &gate, &block).unwrap();
        let expected = [15.090_063f32, 1.581_987_4, 11.0, -1.0];
        for (a, e) in hyper.iter().zip(expected) {
            assert!(close(*a, e), "hyper {a} vs {e}");
        }
    }

    #[test]
    fn read_rejects_wrong_lengths_before_projecting() {
        let norm = [0.0f32, 1.0, 0.0, 0.0];
        let short_norm = [0.0f32, 1.0, 0.0];
        let down = DenseWeight::matrix(1, 4, vec![1.0, 0.5, -1.0, 2.0]);
        let up = DenseWeight::matrix(4, 1, vec![1.0, -1.0, 0.5, 2.0]);
        let inject = DenseWeight::matrix(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]);
        let weights = HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let short_weights = HyperWeights {
            norm: &short_norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let mut p = DenseProjector::default();
        let short_hyper = [3.0f32, 4.0, 1.0];
        assert!(matches!(
            read(&mut p, &shape(), &weights, &short_hyper),
            Err(EngError::BadRequest(_))
        ));
        let hyper = [3.0f32, 4.0, 1.0, 1.0];
        assert!(matches!(
            read(&mut p, &shape(), &short_weights, &hyper),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(p.calls, 0);
    }

    #[test]
    fn apply_rejects_wrong_lengths_without_change() {
        let mut hyper = [3.0f32, 4.0, 1.0, 1.0];
        assert!(matches!(
            apply(&shape(), &mut hyper, &[1.0], &[10.0, -2.0]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(hyper, [3.0, 4.0, 1.0, 1.0]);
        assert!(matches!(
            apply(&shape(), &mut hyper, &[1.0, 1.0], &[1.0, 2.0, 3.0]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(hyper, [3.0, 4.0, 1.0, 1.0]);
    }

    #[test]
    fn read_rejects_projector_output_of_wrong_length() {
        let hyper = [3.0f32, 4.0, 1.0, 1.0];
        let norm = [0.0f32, 1.0, 0.0, 0.0];
        let down = DenseWeight::matrix(2, 4, vec![1.0; 8]);
        let up = DenseWeight::matrix(4, 1, vec![1.0, -1.0, 0.5, 2.0]);
        let inject = DenseWeight::matrix(2, 4, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]);
        let weights = HyperWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let mut p = DenseProjector::default();
        assert!(matches!(
            read(&mut p, &shape(), &weights, &hyper),
            Err(EngError::BadRequest(_))
        ));
    }
}
