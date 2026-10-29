//! Flash-Next MoE decode step (design section "MoE").

use super::math::{dot, expect_len, sigmoid, silu};
use crate::EngError;
use crate::family::project::Projector;

/// Checked MoE dimensions and routing rule.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeShape {
    experts: usize,
    top_k: usize,
    inter: usize,
    shared_inter: usize,
    hidden: usize,
    renorm: bool,
    router_len: usize,
}

impl MoeShape {
    /// Checks the expert count, top-k, intermediate widths, and hidden width.
    pub(crate) fn new(
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
        hidden: u32,
        renorm: bool,
    ) -> Result<Self, EngError> {
        if experts == 0 || top_k == 0 || inter == 0 || shared_inter == 0 || hidden == 0 {
            return Err(EngError::Checkpoint(
                "moe dimensions must exceed zero".into(),
            ));
        }
        if top_k > experts {
            return Err(EngError::Checkpoint(
                "moe top_k exceeds the expert count".into(),
            ));
        }
        let router_len = (experts as usize)
            .checked_mul(hidden as usize)
            .ok_or_else(|| EngError::Checkpoint("moe router size overflows".into()))?;
        Ok(Self {
            experts: experts as usize,
            top_k: top_k as usize,
            inter: inter as usize,
            shared_inter: shared_inter as usize,
            hidden: hidden as usize,
            renorm,
            router_len,
        })
    }

    /// Expert count.
    pub(crate) fn experts(&self) -> usize {
        self.experts
    }
    /// Experts selected per token.
    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }
    /// Whether gates renormalize over the selection.
    pub(crate) fn renorm(&self) -> bool {
        self.renorm
    }
}

/// Weights of one MoE layer.
pub(crate) struct MoeWeights<'a, W> {
    /// Host router matrix `[experts, hidden]`, row-major.
    pub(crate) router: &'a [f32],
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

/// Selects the top `top_k` experts from router probabilities, highest first, the lowest index on
/// ties, with gates renormalized over the selection when `shape.renorm` is set and over all
/// probabilities otherwise. Selection stays on the host.
pub(crate) fn select(shape: &MoeShape, probs: &[f32]) -> Result<Vec<(u32, f32)>, EngError> {
    expect_len(probs, shape.experts, "moe probabilities")?;
    if probs.iter().any(|v| !v.is_finite()) {
        return Err(EngError::Backend {
            operation: "flash_next",
            message: "non-finite router probabilities".into(),
        });
    }
    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(shape.top_k);
    for _ in 0..shape.top_k {
        let mut best: Option<(usize, f32)> = None;
        for (e, &p) in probs.iter().enumerate() {
            if selected.iter().any(|&(s, _)| s == e) {
                continue;
            }
            let replace = match best {
                Some((_, bp)) => p > bp,
                None => true,
            };
            if replace {
                best = Some((e, p));
            }
        }
        match best {
            Some(pick) => selected.push(pick),
            None => return Err(EngError::BadRequest("moe selection is empty".into())),
        }
    }
    let den: f32 = if shape.renorm {
        selected.iter().map(|&(_, p)| p).sum()
    } else {
        probs.iter().sum()
    };
    Ok(selected
        .into_iter()
        .map(|(e, p)| (e as u32, p / den))
        .collect())
}

/// Returns the selected experts in rank order with their gate weights.
pub(crate) fn route(
    shape: &MoeShape,
    router: &[f32],
    x: &[f32],
) -> Result<Vec<(u32, f32)>, EngError> {
    expect_len(x, shape.hidden, "moe input")?;
    expect_len(router, shape.router_len, "moe router")?;
    let logits: Vec<f32> = (0..shape.experts)
        .map(|e| dot(&router[e * shape.hidden..(e + 1) * shape.hidden], x))
        .collect::<Result<_, _>>()?;
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(EngError::Backend {
            operation: "flash_next",
            message: "non-finite router logits".into(),
        });
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    select(shape, &probs)
}

/// Runs one decode step and returns the layer output of length `hidden`.
pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &MoeShape,
    weights: &MoeWeights<'_, P::Weight>,
    x: &[f32],
) -> Result<Vec<f32>, EngError> {
    let routes = route(shape, weights.router, x)?;
    let mut out = vec![0.0f32; shape.hidden];
    for (e, gate) in routes {
        let g = projector.expert_linear(weights.gate, e, x)?;
        expect_len(&g, shape.inter, "moe gate output")?;
        let u = projector.expert_linear(weights.up, e, x)?;
        expect_len(&u, shape.inter, "moe up output")?;
        let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        let o = projector.expert_linear(weights.down, e, &h)?;
        expect_len(&o, shape.hidden, "moe down output")?;
        for (acc, v) in out.iter_mut().zip(&o) {
            *acc += gate * v;
        }
    }
    let sg = projector.linear(weights.shared_gate, x)?;
    expect_len(&sg, shape.shared_inter, "moe shared gate output")?;
    let su = projector.linear(weights.shared_up, x)?;
    expect_len(&su, shape.shared_inter, "moe shared up output")?;
    let sh: Vec<f32> = sg.iter().zip(&su).map(|(a, b)| silu(*a) * b).collect();
    let so = projector.linear(weights.shared_down, &sh)?;
    expect_len(&so, shape.hidden, "moe shared down output")?;
    let gs_raw = projector.linear(weights.shared_scalar, x)?;
    expect_len(&gs_raw, 1, "moe shared scalar")?;
    let gs = sigmoid(gs_raw[0]);
    for (acc, v) in out.iter_mut().zip(&so) {
        *acc += gs * v;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};

    const X: [f32; 2] = [1.0, -1.0];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    struct Fixture {
        router: Vec<f32>,
        gate: DenseWeight,
        up: DenseWeight,
        down: DenseWeight,
        shared_gate: DenseWeight,
        shared_up: DenseWeight,
        shared_down: DenseWeight,
        shared_scalar: DenseWeight,
    }

    fn fixture() -> Fixture {
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..4 {
            let e = e as f32;
            gate.extend([1.0, 0.5 * e, -0.5, 1.0]);
            up.extend([0.5, 1.0, 0.25 * e, -1.0]);
            down.extend([1.0, -1.0, e, 0.5]);
        }
        Fixture {
            router: vec![1.0, -1.0, 0.5, -0.5, 0.5, -0.5, 0.0, 0.0],
            gate: DenseWeight::stack(4, 2, 2, gate),
            up: DenseWeight::stack(4, 2, 2, up),
            down: DenseWeight::stack(4, 2, 2, down),
            shared_gate: DenseWeight::matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            shared_up: DenseWeight::matrix(2, 2, vec![2.0, 0.0, 0.0, -1.0]),
            shared_down: DenseWeight::matrix(2, 2, vec![1.0, 1.0, 1.0, -1.0]),
            shared_scalar: DenseWeight::matrix(1, 2, vec![0.25, -0.75]),
        }
    }

    fn weights(f: &Fixture) -> MoeWeights<'_, DenseWeight> {
        MoeWeights {
            router: &f.router,
            gate: &f.gate,
            up: &f.up,
            down: &f.down,
            shared_gate: &f.shared_gate,
            shared_up: &f.shared_up,
            shared_down: &f.shared_down,
            shared_scalar: &f.shared_scalar,
        }
    }

    fn shape(renorm: bool) -> MoeShape {
        MoeShape::new(4, 2, 2, 2, 2, renorm).unwrap()
    }

    #[test]
    fn shape_rejects_invalid_dimensions() {
        assert!(matches!(
            MoeShape::new(0, 2, 2, 2, 2, true),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            MoeShape::new(4, 5, 2, 2, 2, true),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            MoeShape::new(4, 2, 0, 2, 2, true),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn select_breaks_ties_toward_the_lower_index() {
        let renormed = select(&shape(true), &[0.1, 0.4, 0.4, 0.1]).unwrap();
        assert_eq!(renormed.len(), 2, "selected experts");
        assert_eq!(renormed[0].0, 1, "first expert");
        assert_eq!(renormed[1].0, 2, "second expert");
        assert!(close(renormed[0].1, 0.5), "{} vs 0.5", renormed[0].1);
        assert!(close(renormed[1].1, 0.5), "{} vs 0.5", renormed[1].1);
        let plain = select(&shape(false), &[0.1, 0.4, 0.4, 0.1]).unwrap();
        assert_eq!(plain.len(), 2, "selected experts");
        assert_eq!(plain[0].0, 1, "first expert");
        assert_eq!(plain[1].0, 2, "second expert");
        assert!(close(plain[0].1, 0.4), "{} vs 0.4", plain[0].1);
        assert!(close(plain[1].1, 0.4), "{} vs 0.4", plain[1].1);
        assert!(matches!(
            select(&shape(true), &[0.1, f32::NAN, 0.4, 0.1]),
            Err(EngError::Backend { .. })
        ));
        assert!(matches!(
            select(&shape(true), &[0.5, 0.5]),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn route_breaks_ties_toward_the_lower_index() {
        let f = fixture();
        let routes = route(&shape(true), &f.router, &X).unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].0, 0, "first expert");
        assert_eq!(routes[1].0, 1, "second expert");
        assert!(
            close(routes[0].1, 0.731_058_6),
            "{} vs 0.731_058_6",
            routes[0].1
        );
        assert!(
            close(routes[1].1, 0.268_941_4),
            "{} vs 0.268_941_4",
            routes[1].1
        );
    }

    #[test]
    fn route_without_renorm_divides_by_all_probabilities() {
        let f = fixture();
        let routes = route(&shape(false), &f.router, &X).unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].0, 0, "first expert");
        assert_eq!(routes[1].0, 1, "second expert");
        assert!(
            close(routes[0].1, 0.534_446_6),
            "{} vs 0.534_446_6",
            routes[0].1
        );
        assert!(
            close(routes[1].1, 0.196_611_9),
            "{} vs 0.196_611_9",
            routes[1].1
        );
    }

    #[test]
    fn step_matches_hand_values() {
        let f = fixture();
        let mut p = DenseProjector::default();
        let out = step(&mut p, &shape(true), &weights(&f), &X).unwrap();
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 0.855_243_2), "{} vs 0.855_243_2", out[0]);
        assert!(close(out[1], 1.077_635_7), "{} vs 1.077_635_7", out[1]);
        assert_eq!(p.calls, 10);
    }

    #[test]
    fn step_without_renorm_matches_hand_values() {
        let f = fixture();
        let mut p = DenseProjector::default();
        let out = step(&mut p, &shape(false), &weights(&f), &X).unwrap();
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 0.859_825_5), "{} vs 0.859_825_5", out[0]);
        assert!(close(out[1], 1.128_161_6), "{} vs 1.128_161_6", out[1]);
    }

    #[test]
    fn non_finite_router_is_a_backend_error() {
        let mut f = fixture();
        f.router[0] = f32::NAN;
        let mut p = DenseProjector::default();
        let result = step(&mut p, &shape(true), &weights(&f), &X);
        assert!(matches!(result, Err(EngError::Backend { .. })));
        assert_eq!(p.calls, 0, "projector calls");
    }

    #[test]
    fn wrong_lengths_are_bad_requests() {
        let f = fixture();
        let mut p = DenseProjector::default();
        let long_x = [1.0f32, 2.0, 3.0];
        assert!(matches!(
            step(&mut p, &shape(true), &weights(&f), &long_x),
            Err(EngError::BadRequest(_))
        ));
        let mut short = fixture();
        short.router.truncate(7);
        assert!(matches!(
            step(&mut p, &shape(true), &weights(&short), &X),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn wrong_expert_output_length_is_a_bad_request() {
        let mut f = fixture();
        f.down = DenseWeight::stack(4, 3, 2, vec![1.0; 24]);
        let mut p = DenseProjector::default();
        assert!(matches!(
            step(&mut p, &shape(true), &weights(&f), &X),
            Err(EngError::BadRequest(_))
        ));
    }
}
