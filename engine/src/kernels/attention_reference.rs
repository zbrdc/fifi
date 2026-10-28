//! The f64 oracle of the shared attention contract (mistral.rs study, item 3):
//! exact SDPA with the API's geometry, the tolerance reference for every format
//! cell. Portable pure Rust - it runs on every test build, and the native
//! contract tests (the sink-drop guard, the 513 edge, the per-format tolerances)
//! compare against it. No MLX/CUDA bit-identity claims anywhere.

#[cfg(test)]
pub(crate) mod oracle {
    /// Q [T, Hq, D], K/V [S, Hkv, D], row-major; `gqa = Hq / Hkv`.
    /// Geometry: query row t sits at absolute `query_start + t`, key j at
    /// `key_origin + j`; causal keeps keys `<=` the query position; a window W
    /// (visible keys INCLUDING self) drops keys older than `q_pos - (W - 1)`;
    /// `sinks` are RAW PER QUERY HEAD logits [Hq] (not QK-scaled; GQA shares K/V,
    /// not the softmax denominator offset - Flint: 32 distinct learned values).
    /// Returns [T, Hq, D].
    #[allow(
        clippy::too_many_arguments,
        reason = "the oracle names the full shared-attention geometry - every parameter                     is one axis of the contract it checks"
    )]
    pub(crate) fn sdpa_f64(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        t_rows: usize,
        hq: usize,
        s_len: usize,
        d: usize,
        gqa: usize,
        scale: f64,
        query_start: u32,
        key_origin: u32,
        window: Option<u32>,
        sinks: Option<&[f64]>,
    ) -> Vec<f64> {
        let hkv = hq / gqa;
        let mut out = vec![0.0; t_rows * hq * d];
        for t in 0..t_rows {
            let q_pos = query_start + t as u32;
            for h in 0..hq {
                let kv = h / gqa;
                // True absence: None contributes NO term at all (#3's counterexample -
                // a real QK logit below -1e30 would beat a fabricated -1e30 sink and
                // corrupt the output; a fabricated constant is not generic absence).
                let sink = sinks.map(|s| s[h]);
                // scores over the visible keys, plus the raw sink as a virtual key
                let mut m = sink.unwrap_or(f64::NEG_INFINITY);
                for j in 0..s_len {
                    let k_pos = key_origin + j as u32;
                    let visible = k_pos <= q_pos && window.is_none_or(|w| q_pos - k_pos < w);
                    if !visible {
                        continue;
                    }
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q[(t * hq + h) * d + dd] * k[(j * hkv + kv) * d + dd];
                    }
                    m = m.max(dot * scale);
                }
                let mut denom = sink.map(|sv| (sv - m).exp()).unwrap_or(0.0);
                let mut acc = vec![0.0; d];
                for j in 0..s_len {
                    let k_pos = key_origin + j as u32;
                    let visible = k_pos <= q_pos && window.is_none_or(|w| q_pos - k_pos < w);
                    if !visible {
                        continue;
                    }
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q[(t * hq + h) * d + dd] * k[(j * hkv + kv) * d + dd];
                    }
                    let p = (dot * scale - m).exp();
                    denom += p;
                    for dd in 0..d {
                        acc[dd] += p * v[(j * hkv + kv) * d + dd];
                    }
                }
                // Empty-row guard: no sink and no visible key leaves denom 0 (the row is
                // fully masked); the convention is a zero output, not a NaN.
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                for dd in 0..d {
                    out[(t * hq + h) * d + dd] = acc[dd] * inv;
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Hand-computed: one head, two keys, a sink that must win the denominator.
        /// q = [1, 0], keys = [[1, 0], [0, 1]], v = [[1, 0], [0, 1]], scale 1.
        /// Query at pos 1 (key 0 at pos 0, key 1 at pos 1 - both causal-visible).
        /// Scores: 1, 0. Without a sink: softmax -> [e/(e+1), 1/(e+1)] -> out
        /// = (e*[1,0] + [0,1])/(e+1). With raw sink 2.0: denom gains e^2 and the
        /// output shrinks toward zero - the sink MUST change the answer.
        #[test]
        fn a_raw_sink_moves_the_output() {
            let q = [1.0, 0.0];
            let k = [1.0, 0.0, 0.0, 1.0];
            let v = [1.0, 0.0, 0.0, 1.0];
            let bare = sdpa_f64(&q, &k, &v, 1, 1, 2, 2, 1, 1.0, 1, 0, None, None);
            let sunk = sdpa_f64(&q, &k, &v, 1, 1, 2, 2, 1, 1.0, 1, 0, None, Some(&[2.0]));
            let e = std::f64::consts::E;
            assert!((bare[0] - e / (e + 1.0)).abs() < 1.0e-12);
            assert!(bare[1] > 1.0e-3);
            // the sink mass e^2 dominates: the output collapses toward zero
            assert!(sunk[0] < bare[0] && sunk[1] < bare[1]);
            let sink_denom = e + 1.0 + e * e;
            assert!((sunk[0] - e / sink_denom).abs() < 1.0e-12);
        }

        /// #3's counterexample, closed form: ONE visible key whose real QK logit is
        /// more negative than any fabricated "absent" constant. q=k=F16 extremes with
        /// a huge-but-valid scale give a logit of -2^101 (~-2.5e30): a -1e30 fabricated
        /// sink would win the max and drive the output to 0, but NO-sink truth is
        /// softmax([x]) = 1 -> the output IS v. True absence must return v exactly.
        #[test]
        fn none_is_exact_absence_at_extreme_negative_logits() {
            // one head, one key, one visible position; craft dot*scale = -2^101
            let q = [1.0f64];
            let k = [1.0f64];
            let scale = -(2.0f64).powi(101); // dot 1 * scale -> the extreme logit
            let v = [1.0f64];
            let out = sdpa_f64(&q, &k, &v, 1, 1, 1, 1, 1, scale, 0, 0, None, None);
            assert!(
                (out[0] - 1.0).abs() < 1.0e-12,
                "no-sink softmax of one key is exactly 1 -> v; got {}",
                out[0]
            );
        }

        /// Per-query-head sinks: identical Q/K/V across a GQA group but DIFFERENT
        /// sinks per head MUST give different outputs - a KV-head fold fails this.
        /// Two query heads share one KV head (gqa 2), same q/k/v, sinks [0.0, 3.0]:
        /// head 1's denominator gains e^3 and its output collapses toward zero;
        /// head 0 barely moves. A fold (sinks[kv]) would make both heads equal.
        #[test]
        fn sinks_differ_within_a_gqa_group() {
            let q = [1.0, 0.0, 1.0, 0.0];
            let k = [1.0, 0.0, 0.0, 1.0];
            let v = [1.0, 0.0, 0.0, 1.0];
            let out = sdpa_f64(
                &q,
                &k,
                &v,
                1,
                2,
                2,
                2,
                2,
                1.0,
                1,
                0,
                None,
                Some(&[0.0, 3.0]),
            );
            // head 0: sink 0 -> denom = e + 1 + 1 (scores 1,0; sink 0 adds 1)
            let e = std::f64::consts::E;
            let d0 = e + 1.0 + 1.0;
            assert!((out[0] - e / d0).abs() < 1.0e-12, "head 0 closed form");
            // head 1: sink 3 -> denom = e + 1 + e^3 (dominates) -> output ~ e/denom
            let d1 = e + 1.0 + e * e * e;
            assert!((out[2] - e / d1).abs() < 1.0e-12, "head 1 closed form");
            // the heads DIFFER: a KV-fold would make them equal
            assert!(
                (out[0] - out[2]).abs() > 0.05,
                "per-head sinks must not fold"
            );
        }

        /// W counts visible keys INCLUDING self: at W = 2 a query at pos 3 sees
        /// keys 2 and 3 only; the key at pos 1 is masked even though causal-visible.
        #[test]
        fn the_window_counts_self() {
            // three keys at pos 0,1,2; query at pos 2
            let q = [1.0, 1.0];
            let k = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
            let v = [1.0, 0.0, 2.0, 0.0, 4.0, 0.0];
            let w2 = sdpa_f64(&q, &k, &v, 1, 1, 3, 2, 1, 1.0, 2, 0, Some(2), None);
            // only key pos 2 (score 1, v 4) and pos 1 masked? W=2 -> keys 1,2 visible
            // scores 1,1 -> mean of v rows 1 and 2 -> (2+4)/2 = 3 on element 0.
            assert!((w2[0] - 3.0).abs() < 1.0e-12);
            // W = 3 -> keys 0,1,2 -> mean of 1,2,4 -> 7/3.
            let w3 = sdpa_f64(&q, &k, &v, 1, 1, 3, 2, 1, 1.0, 2, 0, Some(3), None);
            assert!((w3[0] - 7.0 / 3.0).abs() < 1.0e-12);
        }
    }

    /// The bf16 ULP at |x|, stated per the adoption convention: normals use
    /// 2^(floor(log2|x|) - 7); zero and subnormals use the smallest subnormal step
    /// 2^-133 (bf16: 8 exponent bits, 7 mantissa bits).
    pub(crate) fn ulp_bf16(x: f64) -> f64 {
        let a = x.abs();
        if a < 2f64.powi(-126) {
            return 2f64.powi(-133);
        }
        let e = a.log2().floor() as i32;
        2f64.powi(e - 7)
    }

    /// Coordinator-ruled row-scaled metric, shared by host and native tests.
    /// Error uses the f64 reference without an intermediate F32 cast. The scale
    /// is max(|reference element|, maximum |reference| over that head/token row).
    /// Nonfinite values fail before reduction; every element contributes.
    #[allow(clippy::type_complexity)]
    pub(crate) fn pinned_max_ratio(
        got: &[f32],
        want: &[f64], // oracle row-major [T, Hq, D]
        hq: usize,
        t: usize,
        d: usize,
    ) -> (f64, (usize, usize, usize, f32, f64, f64, f64)) {
        let mut worst = 0.0f64;
        let mut offender = (0usize, 0usize, 0usize, 0.0f32, 0.0f64, 0.0f64, 0.0f64);
        for hh in 0..hq {
            for tt in 0..t {
                let mut rowmax = 0.0f64;
                for c in 0..d {
                    let w = want[(tt * hq + hh) * d + c];
                    assert!(w.is_finite(), "nonfinite ref h={hh} t={tt} c={c}");
                    rowmax = rowmax.max(w.abs());
                }
                for c in 0..d {
                    let g = got[(hh * t + tt) * d + c]; // native [1,Hq,T,D]
                    let w = want[(tt * hq + hh) * d + c];
                    assert!(g.is_finite(), "nonfinite got h={hh} t={tt} c={c}");
                    let err = (g as f64 - w).abs();
                    let scale = w.abs().max(rowmax);
                    let ratio = err / ulp_bf16(scale);
                    if ratio > worst {
                        worst = ratio;
                        offender = (hh, tt, c, g, w, rowmax, scale);
                    }
                }
            }
        }
        (worst, offender)
    }

    /// The row maximum determines the scale even for a cancelled element.
    #[test]
    fn pinned_metric_cpu_self_check() {
        let got = vec![1.0f32 + 2f32.powi(-7), 2f32.powi(-10) + 2f32.powi(-12)];
        let want = vec![1.0f64, 2f64.powi(-10)];
        let (worst, off) = pinned_max_ratio(&got, &want, 1, 1, 2);
        // THE COORDINATOR RULING (replaces the 2^-7 floor): scale =
        // max(|ref_elem|, rowmax). Element 0: err 2^-7, scale 1, ulp 2^-7,
        // ratio 1. Element 1: err 2^-12, scale max(2^-10, 1) = 1, ulp 2^-7,
        // ratio 2^-5. MAX over every element = 1 - the rounding tracks the
        // ROW magnitude, not a cancelled element (the ruling's reason).
        assert!((worst - 1.0).abs() < 1e-9, "self-check: {worst}");
        assert_eq!(off.2, 0, "the offender is the large-magnitude element");
    }
}
