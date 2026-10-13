//! Host-side router-guess arithmetic for expert prefetch advice (OFF-2 P9).
//!
//! WHAT: pure arithmetic that guesses the router decision of one MoE layer from
//! an earlier layer's hidden state, `logits = R_L . (rmsnorm(h) * w_L)`, where
//! `R_L` is the layer's router matrix, `w_L` its post-attention-norm weight and
//! `h` the residual after attention of an earlier layer (OFF-3 variant B = L-1,
//! B2/B3 = L-2/L-3). WHY: OFF-3 measured that this recalls 77 percent of the
//! true top-8 with variant B (69 with B2, 65 with B3), which is enough to
//! prefetch expert weights before the real router runs. The guess is advice for
//! prefetching only and can never affect results, so this module is pure host
//! arithmetic: no I/O, no threads, no clock, no prefetch queues, no CUDA. The
//! norm matches the engine's own rmsnorm and the dot uses the fixed 8-lane
//! reduction order of the CPU kernels, so the output bits depend on nothing but
//! the inputs. Staged for later integration (OFF-2 C6/C7).

use std::sync::Arc;

/// One layer's guess inputs: the router matrix and the post-attention-norm
/// weight, both shared and immutable.
pub struct LayerRouter {
    /// The router matrix of the layer, `[experts][hidden]` f32, row-major (ggml
    /// ne0 = hidden fastest), shared because the guess only reads it.
    pub router: Arc<[f32]>,
    /// The post-attention-norm weight of the same layer, `hidden` f32 values.
    pub norm_weight: Arc<[f32]>,
}

/// A validation failure of a guess input, reported before any output is written.
#[derive(Debug, PartialEq)]
pub enum GuessError {
    /// A shape, range or finiteness invariant is broken; the string names it.
    BadInput(String),
}

/// The router-guess arithmetic of a whole model: one [`LayerRouter`] per MoE
/// layer plus the shared geometry. Validate once with [`RouterGuess::new`], then
/// call [`RouterGuess::logits`] per token and layer.
pub struct RouterGuess {
    /// Per-layer router matrix and norm weight, indexed like the model layers.
    layers: Vec<LayerRouter>,
    /// Hidden size: the length of every `h`, norm weight and router row.
    hidden: usize,
    /// Expert count per layer: the router row count and the logits length.
    experts: usize,
    /// The rms-norm epsilon, the model's own `rms_norm_eps` (1e-6 in Qwen3.6).
    eps: f32,
}

impl RouterGuess {
    /// Validates and stores the guess inputs. Every `layers[l].router` must hold
    /// exactly `experts * hidden` finite f32 values and every
    /// `layers[l].norm_weight` exactly `hidden` finite values; `hidden` and
    /// `experts` must be nonzero, `experts` at most 65,536 (the ids the callers
    /// rank are u16); `eps` must be finite and positive. WHY: with every shape
    /// and value checked once, the per-token `logits` call only re-checks the
    /// lengths of the slices it is handed.
    pub fn new(
        layers: Vec<LayerRouter>,
        hidden: usize,
        experts: usize,
        eps: f32,
    ) -> Result<Self, GuessError> {
        if hidden == 0 {
            return Err(GuessError::BadInput(String::from(
                "router guess: hidden must be nonzero",
            )));
        }
        if experts == 0 {
            return Err(GuessError::BadInput(String::from(
                "router guess: experts must be nonzero",
            )));
        }
        if experts > usize::from(u16::MAX) + 1 {
            return Err(GuessError::BadInput(format!(
                "router guess: {experts} experts exceed the 65,536 a u16 id names"
            )));
        }
        let Some(values) = experts.checked_mul(hidden) else {
            return Err(GuessError::BadInput(format!(
                "router guess: {experts} x {hidden} router values overflow usize"
            )));
        };
        for (at, layer) in layers.iter().enumerate() {
            if layer.router.len() != values {
                return Err(GuessError::BadInput(format!(
                    "router guess: layer {at} router holds {} of {values} values",
                    layer.router.len()
                )));
            }
            if layer.norm_weight.len() != hidden {
                return Err(GuessError::BadInput(format!(
                    "router guess: layer {at} norm weight holds {} of {hidden} values",
                    layer.norm_weight.len()
                )));
            }
            if !layer
                .router
                .iter()
                .chain(layer.norm_weight.iter())
                .all(|v| v.is_finite())
            {
                return Err(GuessError::BadInput(format!(
                    "router guess: layer {at} holds a non-finite value"
                )));
            }
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(GuessError::BadInput(format!(
                "router guess: eps {eps} is not finite and positive"
            )));
        }
        Ok(Self {
            layers,
            hidden,
            experts,
            eps,
        })
    }

    /// The layer count, the upper bound `target_layer` is checked against.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The router logits of `target_layer` for the earlier hidden state `h`:
    /// rms-normalise `h` with `target_layer`'s own norm weight (the engine's
    /// formula: f64 mean and scale, per-value cast, weight in f32), then the
    /// row-major matvec against that layer's router. `h` must be `hidden` long
    /// and finite and `out` exactly `experts` long; `out` is fully overwritten
    /// and stays untouched on error. The dot fills 8 lane accumulators in index
    /// order, so the bits never depend on anything but the inputs.
    pub fn logits(
        &self,
        target_layer: usize,
        h: &[f32],
        out: &mut [f32],
    ) -> Result<(), GuessError> {
        let Some(layer) = self.layers.get(target_layer) else {
            return Err(GuessError::BadInput(format!(
                "router guess: target layer {target_layer} of {} layers",
                self.layers.len()
            )));
        };
        if h.len() != self.hidden {
            return Err(GuessError::BadInput(format!(
                "router guess: hidden state holds {} of {} values",
                h.len(),
                self.hidden
            )));
        }
        if out.len() != self.experts {
            return Err(GuessError::BadInput(format!(
                "router guess: logits out holds {} of {} values",
                out.len(),
                self.experts
            )));
        }
        if !h.iter().all(|v| v.is_finite()) {
            return Err(GuessError::BadInput(String::from(
                "router guess: hidden state holds a non-finite value",
            )));
        }
        let squares = h.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
        let scale = 1.0 / (squares / self.hidden as f64 + f64::from(self.eps)).sqrt();
        let normed: Vec<f32> = h
            .iter()
            .zip(layer.norm_weight.iter())
            .map(|(v, w)| ((f64::from(*v) * scale) as f32) * w)
            .collect();
        for (o, row) in out.iter_mut().zip(layer.router.chunks_exact(self.hidden)) {
            *o = dot8(row, &normed);
        }
        Ok(())
    }

    /// The ids of the `k` largest logits of `target_layer` for the earlier hidden
    /// state `h`, best logit first; equal logits keep the LOWER id first, `k == 0`
    /// returns an empty list and `k > experts` is refused. The ranking orders by
    /// `total_cmp`, a total order, so even a NaN logit (impossible with finite
    /// inputs) can neither panic the sort nor reorder two runs of one input. WHY:
    /// these ids are the prefetch advice itself, `u16` like every expert id.
    pub fn top_k(&self, target_layer: usize, h: &[f32], k: usize) -> Result<Vec<u16>, GuessError> {
        if k > self.experts {
            return Err(GuessError::BadInput(format!(
                "router guess: top {k} of {experts} experts",
                experts = self.experts
            )));
        }
        let mut logits = vec![0.0f32; self.experts];
        self.logits(target_layer, h, &mut logits)?;
        let mut ranked: Vec<usize> = (0..self.experts).collect();
        ranked.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        ranked.truncate(k);
        ranked
            .into_iter()
            .map(|id| {
                u16::try_from(id).map_err(|_| {
                    GuessError::BadInput(format!("router guess: id {id} does not fit u16"))
                })
            })
            .collect()
    }
}

/// How many of the `actual` experts the `guess` contains, `|guess ∩ actual|`; a
/// duplicate in either list counts once. WHY: recall of the advice against the
/// true routing is the quality number of the guess, and a repeat must not buy hits.
pub fn hits(guess: &[u16], actual: &[u16]) -> usize {
    let mut count = 0;
    for (at, id) in guess.iter().enumerate() {
        if !guess[..at].contains(id) && actual.contains(id) {
            count += 1;
        }
    }
    count
}

/// Running recall of the advice: `add` one (guess, actual) pair per layer and
/// step, and `recall()` is the hit fraction of all actual experts seen. WHY: a
/// running number reports prefetch quality without keeping the pairs around.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct RecallMeter {
    /// The running sum of [`hits`] over every `add`.
    hits: u64,
    /// The running sum of `actual.len()` over every `add`.
    total: u64,
    /// How many (guess, actual) pairs were added.
    samples: u64,
}

impl RecallMeter {
    /// Adds one layer's step: [`hits`]` (guess, actual)` hits of `actual.len()`
    /// actual experts. WHY: one layer per call keeps the meter free of any
    /// layering assumption; call it once per (step, layer) pair.
    pub fn add(&mut self, guess: &[u16], actual: &[u16]) {
        self.hits += hits(guess, actual) as u64;
        self.total += actual.len() as u64;
        self.samples += 1;
    }

    /// `hits / total`, 0.0 when nothing was added (so averages never see NaN).
    pub fn recall(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f64 / self.total as f64
        }
    }

    /// How many (guess, actual) pairs were added so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }
}

/// Sums the 8 lane accumulators as `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`; that
/// fixed order is this module's determinism contract, as in the CPU kernels.
fn reduce8(l: [f32; 8]) -> f32 {
    (l[0] + l[1]) + (l[2] + l[3]) + (l[4] + l[5]) + (l[6] + l[7])
}

/// The f32 dot of two equal-length slices: value `i` accumulates into lane
/// `i % 8` in increasing `i` and the lanes reduce once at the end. WHY: with a
/// fixed lane order the sum cannot depend on the compiler's vector width, so
/// identical inputs give bit-identical logits.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (chunks, tail) = a.as_chunks::<8>();
    let (peer_chunks, peer_tail) = b.as_chunks::<8>();
    for (ca, cb) in chunks.iter().zip(peer_chunks) {
        for l in 0..8 {
            lanes[l] += ca[l] * cb[l];
        }
    }
    for (l, (x, y)) in tail.iter().zip(peer_tail).enumerate() {
        lanes[l] += x * y;
    }
    reduce8(lanes)
}

#[cfg(test)]
mod tests {
    use super::{GuessError, LayerRouter, RecallMeter, RouterGuess, hits};
    use std::sync::Arc;

    /// The tiny deterministic LCG of the offline tests (rule 5): a fixed seed
    /// and the Knuth MMIX constants, no OS randomness.
    struct Lcg(u64);

    impl Lcg {
        /// One step; the low bits of an LCG are weak, so callers take the top.
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        /// A uniform f32 in `[lo, hi)`.
        fn f32_between(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
        }
    }

    /// One random layer of a guess: router values at `1 / sqrt(hidden)` scale
    /// and norm weights around 1, checkpoint magnitudes with moderate logits.
    fn random_layer(rng: &mut Lcg, hidden: usize, experts: usize) -> LayerRouter {
        let norm = 1.0f32 / (hidden as f32).sqrt();
        LayerRouter {
            router: (0..experts * hidden)
                .map(|_| rng.f32_between(-1.0, 1.0) * norm)
                .collect::<Vec<_>>()
                .into(),
            norm_weight: (0..hidden)
                .map(|_| rng.f32_between(0.5, 1.5))
                .collect::<Vec<_>>()
                .into(),
        }
    }

    /// A guess of `layers` layers of random weights with the model's 1e-6 eps.
    fn random_guess(seed: u64, layers: usize, hidden: usize, experts: usize) -> RouterGuess {
        let mut rng = Lcg(seed);
        let made = (0..layers)
            .map(|_| random_layer(&mut rng, hidden, experts))
            .collect();
        RouterGuess::new(made, hidden, experts, 1e-6).unwrap()
    }

    #[test]
    fn logits_match_a_double_precision_reference() {
        // Per-logit bound: |got - ref| <= 1e-4 * sum(|term|) + 1e-6, the stated
        // tolerance of the C5 kernel tests, where ref is an independent f64
        // pipeline: the same rmsnorm in f64 and a plain f64 matvec.
        let (hidden, experts) = (2048usize, 256usize);
        let mut rng = Lcg(0x0009_0900_0000_0001);
        let scale = 1.0f32 / (hidden as f32).sqrt();
        let router: Vec<f32> = (0..experts * hidden)
            .map(|_| rng.f32_between(-1.0, 1.0) * scale)
            .collect();
        let norm: Vec<f32> = (0..hidden).map(|_| rng.f32_between(0.5, 1.5)).collect();
        let h: Vec<f32> = (0..hidden).map(|_| rng.f32_between(-2.0, 2.0)).collect();
        let guess = RouterGuess::new(
            vec![LayerRouter {
                router: router.clone().into(),
                norm_weight: norm.clone().into(),
            }],
            hidden,
            experts,
            1e-6,
        )
        .unwrap();
        let mut got = vec![0.0f32; experts];
        guess.logits(0, &h, &mut got).unwrap();

        let mean = h.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / hidden as f64;
        let inv = 1.0 / (mean + 1e-6).sqrt();
        let v: Vec<f64> = h
            .iter()
            .zip(&norm)
            .map(|(x, w)| f64::from(*x) * inv * f64::from(*w))
            .collect();
        for (expert, out) in got.iter().enumerate() {
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            for (column, term) in v.iter().enumerate() {
                let t = f64::from(router[expert * hidden + column]) * term;
                sum += t;
                abs += t.abs();
            }
            let bound = 1e-4 * abs + 1e-6;
            assert!(
                (f64::from(*out) - sum).abs() <= bound,
                "expert {expert}: got {out}, reference {sum}, bound {bound}"
            );
        }
    }

    #[test]
    fn the_guess_uses_the_target_layers_own_norm_weight() {
        // Two layers with one shared router but two different norm weights: the
        // logits must differ, and swapping the weights swaps the results bitwise.
        // This is the +9-10 recall points of OFF-3 variant C over D.
        let (hidden, experts) = (64usize, 6usize);
        let mut rng = Lcg(0x0009_0900_0000_0002);
        let norm = 1.0f32 / (hidden as f32).sqrt();
        let shared: Arc<[f32]> = (0..experts * hidden)
            .map(|_| rng.f32_between(-1.0, 1.0) * norm)
            .collect::<Vec<_>>()
            .into();
        let a: Arc<[f32]> = (0..hidden)
            .map(|_| rng.f32_between(0.5, 1.5))
            .collect::<Vec<_>>()
            .into();
        let b: Arc<[f32]> = (0..hidden)
            .map(|_| rng.f32_between(0.5, 1.5))
            .collect::<Vec<_>>()
            .into();
        let h: Vec<f32> = (0..hidden).map(|_| rng.f32_between(-1.0, 1.0)).collect();
        let layer_of = |norm_weight: &Arc<[f32]>| LayerRouter {
            router: Arc::clone(&shared),
            norm_weight: Arc::clone(norm_weight),
        };
        let bits = |guess: &RouterGuess, layer: usize, h: &[f32]| -> Vec<u32> {
            let mut out = vec![0.0f32; experts];
            guess.logits(layer, h, &mut out).unwrap();
            out.iter().map(|v| v.to_bits()).collect()
        };
        let guess =
            RouterGuess::new(vec![layer_of(&a), layer_of(&b)], hidden, experts, 1e-6).unwrap();
        let (la, lb) = (bits(&guess, 0, &h), bits(&guess, 1, &h));
        assert_ne!(la, lb, "one router, two norm weights: logits must differ");
        let swapped =
            RouterGuess::new(vec![layer_of(&b), layer_of(&a)], hidden, experts, 1e-6).unwrap();
        assert_eq!(bits(&swapped, 0, &h), lb);
        assert_eq!(bits(&swapped, 1, &h), la);
    }

    #[test]
    fn logits_are_bitwise_deterministic_and_independent_of_buffer_contents() {
        let guess = random_guess(0x0009_0900_0000_0003, 2, 128, 32);
        let mut rng = Lcg(0x0009_0900_0000_0004);
        let h: Vec<f32> = (0..128).map(|_| rng.f32_between(-1.0, 1.0)).collect();
        let mut clean = vec![0.0f32; 32];
        guess.logits(1, &h, &mut clean).unwrap();
        for dirty in [vec![f32::NAN; 32], vec![f32::INFINITY; 32]] {
            let mut out = dirty;
            guess.logits(1, &h, &mut out).unwrap();
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                clean.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "the logits must not read the out buffer"
            );
        }
        assert!(clean.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn new_refuses_bad_shapes_and_non_finite_weights() {
        let guess_of =
            |hidden: usize, experts: usize, router: Vec<f32>, norm: Vec<f32>, eps: f32| {
                RouterGuess::new(
                    vec![LayerRouter {
                        router: router.into(),
                        norm_weight: norm.into(),
                    }],
                    hidden,
                    experts,
                    eps,
                )
            };
        let shape = |hidden: usize, experts: usize| {
            guess_of(
                hidden,
                experts,
                vec![0.25; experts * hidden],
                vec![1.0; hidden],
                1e-6,
            )
        };
        // Shapes: nonzero dims, experts within the u16 ids, exact slice lengths.
        for (hidden, experts, what) in [
            (0, 3, "hidden 0"),
            (4, 0, "experts 0"),
            (4, 65_537, "experts above the 65,536 u16 ids"),
        ] {
            assert!(shape(hidden, experts).is_err(), "{what} must be refused");
        }
        assert!(shape(4, 3).is_ok());
        assert!(
            shape(4, 65_536).is_ok(),
            "65,536 experts is the u16 id limit"
        );
        for (router_len, norm_len, what) in [
            (11, 4, "router short"),
            (13, 4, "router long"),
            (12, 3, "norm short"),
            (12, 5, "norm long"),
        ] {
            assert!(
                guess_of(4, 3, vec![0.25; router_len], vec![1.0; norm_len], 1e-6).is_err(),
                "{what} must be refused"
            );
        }
        let mut nan_router = vec![0.25f32; 12];
        nan_router[7] = f32::NAN;
        let mut inf_norm = vec![1.0f32; 4];
        inf_norm[2] = f32::NEG_INFINITY;
        assert!(
            guess_of(4, 3, nan_router, vec![1.0; 4], 1e-6).is_err(),
            "router NaN"
        );
        assert!(
            guess_of(4, 3, vec![0.25; 12], inf_norm, 1e-6).is_err(),
            "norm -inf"
        );
        for eps in [0.0f32, -1e-6, f32::NAN, f32::INFINITY] {
            assert!(
                guess_of(4, 3, vec![0.25; 12], vec![1.0; 4], eps).is_err(),
                "eps {eps}"
            );
        }
        // An empty layer list is legal: every call refuses its target layer, so
        // a guess built before the weights arrive cannot panic.
        let empty = RouterGuess::new(vec![], 4, 3, 1e-6).unwrap();
        assert_eq!(empty.layers(), 0);
        assert!(empty.logits(0, &[0.0; 4], &mut [0.0; 3]).is_err());
    }

    #[test]
    fn logits_refuse_a_wrong_length_or_non_finite_hidden_state() {
        let guess = random_guess(0x0009_0900_0000_0005, 1, 8, 4);
        let h = vec![0.5f32; 8];
        let mut out = vec![f32::NAN; 4];
        let mut wrong = vec![f32::NAN; 5];
        let mut nan = h.clone();
        nan[3] = f32::NAN;
        assert!(guess.logits(1, &h, &mut out).is_err(), "layer out of range");
        assert!(guess.logits(0, &h[..7], &mut out).is_err(), "h short");
        assert!(guess.logits(0, &nan, &mut out).is_err(), "h NaN");
        assert!(guess.logits(0, &h, &mut wrong).is_err(), "out wrong length");
        assert!(
            out.iter().all(|v| v.is_nan()) && wrong.iter().all(|v| v.is_nan()),
            "a refused call must not touch out"
        );
        guess.logits(0, &h, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "a valid call fills out");
    }

    #[test]
    fn top_k_orders_by_logit_and_breaks_ties_by_lower_id() {
        // hidden 2, experts 4: h = [1, 0] makes rmsnorm(h) * w = [s, 0], so the
        // logit of row e is router[e][0] * s: rows 0 and 1 tie, 2 doubles, 3 is 0.
        let router = [[1.0f32, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]].concat();
        let guess = RouterGuess::new(
            vec![LayerRouter {
                router: router.into(),
                norm_weight: vec![1.0, 1.0].into(),
            }],
            2,
            4,
            1e-6,
        )
        .unwrap();
        assert_eq!(guess.top_k(0, &[1.0, 0.0], 4).unwrap(), vec![2, 0, 1, 3]);
        assert_eq!(
            guess.top_k(0, &[1.0, 0.0], 2).unwrap(),
            vec![2, 0],
            "the equal logits of 0 and 1 keep the lower id"
        );
    }

    #[test]
    fn top_k_of_zero_is_empty_and_k_above_experts_is_refused() {
        let guess = random_guess(0x0009_0900_0000_0006, 1, 8, 4);
        let h = vec![0.25f32; 8];
        assert_eq!(guess.top_k(0, &h, 0).unwrap(), Vec::<u16>::new());
        assert_eq!(
            guess.top_k(0, &h, 5),
            Err(GuessError::BadInput(String::from(
                "router guess: top 5 of 4 experts"
            )))
        );
        let ids = guess.top_k(0, &h, 4).unwrap();
        assert_eq!(ids.len(), 4, "k == experts is the full ranking");
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "every id appears once");
    }

    #[test]
    fn a_planted_router_recovers_its_planted_experts() {
        let (hidden, experts) = (2048usize, 256usize);
        let mut rng = Lcg(0x0009_0900_0000_0007);
        let h: Vec<f32> = (0..hidden).map(|_| rng.f32_between(-1.0, 1.0)).collect();
        let norm: Vec<f32> = (0..hidden).map(|_| rng.f32_between(0.5, 1.5)).collect();
        // Every row is small noise except rows 17, 99 and 3, which point along h
        // with three strengths: their logits are positive multiples of sum(h^2*w).
        let tiny = 0.05f32 / (hidden as f32).sqrt();
        let mut router: Vec<f32> = (0..experts * hidden)
            .map(|_| rng.f32_between(-1.0, 1.0) * tiny)
            .collect();
        for &(id, strength) in &[(17usize, 1.3f32), (99usize, 1.15f32), (3usize, 1.0f32)] {
            for (r, hv) in router[id * hidden..(id + 1) * hidden].iter_mut().zip(&h) {
                *r = strength * hv;
            }
        }
        let layer = LayerRouter {
            router: router.into(),
            norm_weight: norm.into(),
        };
        let guess = RouterGuess::new(vec![layer], hidden, experts, 1e-6).unwrap();
        let top = guess.top_k(0, &h, 3).unwrap();
        assert_eq!(top, vec![17, 99, 3], "the planted rows win in logit order");
        // End to end: a meter fed the advice against the planted truth reads 1.0.
        let mut meter = RecallMeter::default();
        meter.add(&top, &[17, 99, 3]);
        meter.add(&guess.top_k(0, &h, 1).unwrap(), &[17]);
        assert_eq!(meter.recall(), 1.0);
        assert_eq!(meter.samples(), 2);
    }

    #[test]
    fn hits_counts_the_intersection_once() {
        assert_eq!(hits(&[1, 2, 2], &[2, 1, 1]), 2, "either side may repeat");
        assert_eq!(hits(&[7, 7, 7], &[7]), 1, "a tripled guess is one hit");
        assert_eq!(hits(&[3], &[9, 4]), 0);
        assert_eq!(hits(&[], &[1, 2]), 0);
        assert_eq!(hits(&[1, 2], &[]), 0);
    }

    #[test]
    fn recall_meter_accumulates_over_steps() {
        let mut meter = RecallMeter::default();
        assert_eq!((meter.recall(), meter.samples()), (0.0, 0));
        meter.add(&[1, 2, 3], &[2, 3, 4]);
        meter.add(&[5, 5], &[5]);
        assert_eq!(meter.recall(), 3.0 / 4.0, "2 of 3, then 1 of 1");
        assert_eq!(meter.samples(), 2);
        meter.add(&[], &[]);
        assert_eq!(meter.recall(), 3.0 / 4.0, "an empty layer adds nothing");
        assert_eq!(meter.samples(), 3);
    }
}
