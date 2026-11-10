//! Test-only scalar references for gated-delta prefill.

#[derive(Clone, Copy)]
pub(crate) enum Norm {
    Ggml,
    AddEps,
    QwenRms,
}

pub(crate) fn normalize(x: &[f32], eps: f64, scale: f64, norm: Norm) -> Vec<f64> {
    let sum: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
    let denominator = match norm {
        Norm::Ggml => sum.max(eps * eps),
        Norm::AddEps => sum + eps,
        Norm::QwenRms => sum + x.len() as f64 * eps,
    }
    .sqrt();
    x.iter()
        .map(|&v| f64::from(v) * scale / denominator)
        .collect()
}

/// State is `[key_dim, value_dim]`, with the value dimension contiguous.
pub(crate) fn step(
    state: &mut [f64],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: f64,
    beta: f64,
) -> Vec<f64> {
    assert_eq!(q.len(), k.len());
    assert_eq!(state.len(), k.len() * v.len());
    let mut sk = vec![0.0; v.len()];
    for i in 0..k.len() {
        for j in 0..v.len() {
            state[i * v.len() + j] *= decay;
            sk[j] += state[i * v.len() + j] * k[i];
        }
    }
    let delta: Vec<f64> = (0..v.len()).map(|j| (v[j] - sk[j]) * beta).collect();
    let mut out = vec![0.0; v.len()];
    for i in 0..k.len() {
        for j in 0..v.len() {
            state[i * v.len() + j] += k[i] * delta[j];
            out[j] += state[i * v.len() + j] * q[i];
        }
    }
    out
}

/// The chunk-parallel (WY) form of the same recurrence, in f64: `c` tokens of one head
/// reorganized so the token loop carries no per-token D×D state product. With `B_t` the
/// prefix product of the decays and `G[t][s] = beta_t·(B_t/B_s)·(k_t·k_s)` for `s < t`:
///
/// - `u_t = beta_t·(v_t − B_t·Sᵀk_t) − Σ_{s<t} G[t][s]·u_s` (forward substitution),
/// - `S ← B_{c−1}·S + Σ_s (B_{c−1}/B_s)·k_s u_sᵀ`,
/// - `o_t = B_t·(q_tᵀS) + Σ_{s≤t} (B_t/B_s)·(q_tᵀk_s)·u_s` (against the sub-chunk's
///   START state `S`),
///
/// which is the scalar recurrence of [`step`] unrolled (decay first, delta against the
/// decayed state, output against the post-update state). Pinned against `step` by
/// `chunk_scan_matches_the_step_reference` at f64 exactness; the CUDA kernel
/// `gdn_scan_chunk_k` is this function in fp32.
pub(crate) fn chunk_step(
    state: &mut [f64],
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    decay: &[f64],
    beta: &[f64],
) -> Vec<Vec<f64>> {
    let c = v.len();
    let d = k[0].len();
    let dv = v[0].len();
    assert_eq!(q.len(), c);
    assert_eq!(state.len(), d * dv);
    // The prefix products B_t; the ratios below are divisions in f64 (the kernel computes
    // them as running products because fp32 B_t underflows long before the ratio does).
    let mut prefix = vec![1.0_f64; c];
    for t in 0..c {
        prefix[t] = if t == 0 {
            decay[0]
        } else {
            prefix[t - 1] * decay[t]
        };
    }
    let kk = |t: usize, s: usize| -> f64 { (0..d).map(|r| k[t][r] * k[s][r]).sum::<f64>() };
    let mut u = vec![vec![0.0_f64; dv]; c];
    for t in 0..c {
        // w_t = B_t·(Sᵀk_t), the sub-chunk's start state seen by token t.
        let w: Vec<f64> = (0..dv)
            .map(|j| prefix[t] * (0..d).map(|r| k[t][r] * state[r * dv + j]).sum::<f64>())
            .collect();
        for j in 0..dv {
            let back: f64 = (0..t)
                .map(|s| beta[t] * (prefix[t] / prefix[s]) * kk(t, s) * u[s][j])
                .sum();
            u[t][j] = beta[t] * (v[t][j] - w[j]) - back;
        }
    }
    let out: Vec<Vec<f64>> = (0..c)
        .map(|t| {
            (0..dv)
                .map(|j| {
                    let inter =
                        prefix[t] * (0..d).map(|r| q[t][r] * state[r * dv + j]).sum::<f64>();
                    let intra: f64 = (0..=t)
                        .map(|s| {
                            let qq = (0..d).map(|r| q[t][r] * k[s][r]).sum::<f64>();
                            (prefix[t] / prefix[s]) * qq * u[s][j]
                        })
                        .sum();
                    inter + intra
                })
                .collect()
        })
        .collect();
    for r in 0..d {
        for j in 0..dv {
            let add: f64 = (0..c)
                .map(|s| (prefix[c - 1] / prefix[s]) * k[s][r] * u[s][j])
                .sum();
            state[r * dv + j] = prefix[c - 1] * state[r * dv + j] + add;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Norm, chunk_step, normalize, step};

    /// The xorshift Rng of `cuda/testkit.rs`, restated because `kernels::cuda` is
    /// feature-gated and this module's tests run host-only.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            })
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
        }
    }

    #[test]
    fn normalization_forms_are_distinct_for_sparse_near_zero_input() {
        let mut x = [0.0f32; 128];
        x[0] = 1e-5;
        let expected = [
            (Norm::Ggml, 1.0),
            (Norm::AddEps, 0.0099995),
            (Norm::QwenRms, 0.000883883),
        ];
        for (form, want) in expected {
            let out = normalize(&x, 1e-6, 1.0, form);
            assert!((out[0] - want).abs() < 1e-8);
            assert!(out[1..].iter().all(|&v| v == 0.0));
            assert!(
                normalize(&[0.0; 128], 1e-6, 1.0, form)
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }
    }

    #[test]
    fn chunk_scan_matches_the_step_reference() {
        // Both forms in f64 over adversarial shapes: c = 1, c = m, a ragged split, key
        // vectors at the near-parallel extreme (kk close to 1 stresses the solve) and
        // near-orthogonal, decays down to 0.06 and betas across (0,1).
        struct Case {
            d: usize,
            dv: usize,
            tokens: usize,
            chunk: usize,
            seed: u64,
        }
        let cases = [
            Case {
                d: 8,
                dv: 5,
                tokens: 17,
                chunk: 4,
                seed: 1,
            },
            Case {
                d: 16,
                dv: 16,
                tokens: 33,
                chunk: 32,
                seed: 2,
            },
            Case {
                d: 12,
                dv: 7,
                tokens: 9,
                chunk: 9,
                seed: 3,
            },
            Case {
                d: 32,
                dv: 3,
                tokens: 65,
                chunk: 1,
                seed: 4,
            },
        ];
        for case in cases {
            let mut rng = Rng::new(case.seed);
            let draw = |rng: &mut Rng, n: usize| {
                // Unit rows with a random shared component: kk spans (0, 1).
                let share = rng.unit();
                let mut row: Vec<f64> = (0..n).map(|_| rng.normal() * 0.3 + share).collect();
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                row.iter_mut().for_each(|x| *x /= norm);
                row
            };
            let mut state: Vec<f64> = (0..case.d * case.dv).map(|_| rng.normal() * 0.1).collect();
            let mut seq = state.clone();
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            for _ in 0..case.tokens {
                qs.push(draw(&mut rng, case.d));
                ks.push(draw(&mut rng, case.d));
                vs.push((0..case.dv).map(|_| rng.normal()).collect::<Vec<f64>>());
            }
            let decays: Vec<f64> = (0..case.tokens)
                .map(|_| 0.06_f64.powf(rng.unit()))
                .collect();
            let betas: Vec<f64> = (0..case.tokens).map(|_| rng.unit()).collect();
            // The sequential run.
            let mut seq_out = Vec::new();
            for t in 0..case.tokens {
                let o = step(&mut seq, &qs[t], &ks[t], &vs[t], decays[t], betas[t]);
                seq_out.push(o);
            }
            // The chunked run over the same tokens.
            let mut chunk_out = Vec::new();
            for t0 in (0..case.tokens).step_by(case.chunk) {
                let end = (t0 + case.chunk).min(case.tokens);
                let rows = chunk_step(
                    &mut state,
                    &qs[t0..end],
                    &ks[t0..end],
                    &vs[t0..end],
                    &decays[t0..end],
                    &betas[t0..end],
                );
                chunk_out.extend(rows);
            }
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
            for t in 0..case.tokens {
                for j in 0..case.dv {
                    let gap = rel(seq_out[t][j], chunk_out[t][j]);
                    assert!(
                        gap < 1e-10,
                        "case (d {}, dv {}, c {}): output gap {gap} at t {t} j {j}",
                        case.d,
                        case.dv,
                        case.chunk
                    );
                }
            }
            for (i, (a, b)) in seq.iter().zip(&state).enumerate() {
                let gap = rel(*a, *b);
                assert!(gap < 1e-10, "state gap {gap} at {i}");
            }
        }
    }

    #[test]
    fn recurrence_matches_hand_values_with_nonzero_state() {
        let mut state = [1.0, 3.0, 2.0, 4.0];
        let out = step(&mut state, &[0.0, 1.0], &[1.0, 0.0], &[2.0, 4.0], 0.5, 0.5);
        assert_eq!(state, [1.25, 2.75, 1.0, 2.0]);
        assert_eq!(out, [1.0, 2.0]);
        let out = step(&mut state, &[1.0, 2.0], &[1.0, 1.0], &[1.0, 2.0], 0.25, 0.5);
        assert_eq!(state, [17.0 / 32.0, 35.0 / 32.0, 15.0 / 32.0, 29.0 / 32.0]);
        assert_eq!(out, [47.0 / 32.0, 93.0 / 32.0]);
    }
}
