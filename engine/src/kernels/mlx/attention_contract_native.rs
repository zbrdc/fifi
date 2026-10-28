//! The native contract tests of the shared attention API (the mistral.rs study):
//! the sink-drop guard (MLX 0.32.2's sdpa_vector_2pass_1_gqa has NO sink support -
//! a silently dropped sink makes sinked == sinkless, which FAILS loudly here), the
//! 513 window edge, the per-query-head sink contract on the REAL kernels (a KV fold
//! must fail), and the per-format tolerance cells against the portable f64 oracle.
//! Ignore-gated tiny GPU tests, no model weights - the explicit-allocation class.

use super::*;
use crate::{
    kernels::attention_reference::oracle::{pinned_max_ratio, sdpa_f64, ulp_bf16},
    tensor::{Dtype, Tensor},
};

/// Deterministic pseudo-random f32 in [-1, 1] - no RNG dependency.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn import_f32(k: &mut MlxKernels, shape: Vec<u32>, data: &[f32]) -> ExecutionTensor {
    let host = Tensor::try_from_host(
        Dtype::F32,
        shape,
        data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    k.import_tensor(&host).unwrap()
}

fn readback_f32(k: &mut MlxKernels, t: &ExecutionTensor) -> Vec<f32> {
    let t = k.cast(t, Dtype::F32).unwrap();
    let host = k.readback(&t).unwrap();
    host.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn finite_negative_bf16_logits_have_no_sentinel_cutoff() {
    let _serial = crate::native_test_lock();
    let hi = 2.0f32.powi(127);
    for (key_logit, sink_logit, expected) in [
        (-2.0f32.powi(101), -2.0f32.powi(102), 1.0),
        (-2.0f32.powi(102), -2.0f32.powi(101), 0.0),
        (-2.0f32.powi(101), -2.0f32.powi(101), 0.5),
        (-1.5 * hi, -1.75 * hi, 1.0),
        (-1.5 * hi, -1.5 * hi, 0.5),
    ] {
        assert!(key_logit.is_finite() && sink_logit.is_finite());
        let mut k = MlxKernels::try_new().unwrap();
        k.allow_bf16(true);
        let mut q = vec![0.0; 8 * 64];
        for head in q.as_chunks_mut::<64>().0 {
            head[0] = 1.0;
        }
        let mut keys = vec![0.0; 64];
        keys[0] = key_logit;
        let q = import_f32(&mut k, vec![1, 8, 1, 64], &q);
        let q = k.cast(&q, Dtype::Bf16).unwrap();
        let keys = import_f32(&mut k, vec![1, 1, 1, 64], &keys);
        let keys = k.cast(&keys, Dtype::Bf16).unwrap();
        let values = import_f32(&mut k, vec![1, 1, 1, 64], &[1.0; 64]);
        let values = k.cast(&values, Dtype::Bf16).unwrap();
        let sinks = import_f32(&mut k, vec![1, 8], &[sink_logit; 8]);
        let out = k
            .shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: 0,
                key_origin: 0,
                window: None,
                sinks: Some(&sinks),
                scale: 1.0,
            })
            .unwrap();
        let out = readback_f32(&mut k, &out);
        assert_eq!(out.len(), 8 * 64);
        assert!(
            out.iter().all(|&value| value == expected),
            "finite-negative logits: key={key_logit:e}, sink={sink_logit:e}, expected={expected}, first={}",
            out[0]
        );
    }
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn absent_float_sink_survives_empty_window_tiles() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let mut q = vec![0.0; 8 * 20 * 64];
    for row in q.as_chunks_mut::<64>().0 {
        row[0] = 1.0;
    }
    let mut keys = vec![0.0; 20 * 64];
    for row in keys.as_chunks_mut::<64>().0 {
        row[0] = -2.0f32.powi(101);
    }
    let q = import_f32(&mut k, vec![1, 8, 20, 64], &q);
    let q = k.cast(&q, Dtype::Bf16).unwrap();
    let keys = import_f32(&mut k, vec![1, 1, 20, 64], &keys);
    let keys = k.cast(&keys, Dtype::Bf16).unwrap();
    let values = import_f32(&mut k, vec![1, 1, 20, 64], &[1.0; 20 * 64]);
    let values = k.cast(&values, Dtype::Bf16).unwrap();
    let out = k
        .shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 0,
            key_origin: 0,
            window: Some(2),
            sinks: None,
            scale: 1.0,
        })
        .unwrap();
    let out = readback_f32(&mut k, &out);
    assert_eq!(out.len(), 8 * 20 * 64);
    for (index, value) in out.iter().enumerate() {
        assert_eq!(*value, 1.0, "absent sink and empty window tile at {index}");
    }
}

/// The SINK-DROP GUARD: at k_len > 1024 with GQA (the shape that routes MLX 0.32.2
/// to sdpa_vector_2pass_1_gqa, which has NO sink support), a strong sink MUST
/// change the output - identical outputs mean the sink vanished.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn a_sink_survives_the_long_gqa_float_path() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (8, 1, 1, 64, 1100);
    let mut seed = 0xC0FFEEu64;
    let q: Vec<f32> = (0..hq * t * d).map(|_| lcg(&mut seed)).collect();
    let kv: Vec<f32> = (0..2 * hkv * s * d).map(|_| lcg(&mut seed)).collect();
    let (keys, values) = kv.split_at(hkv * s * d);
    let q_t = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let k_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], keys);
    let v_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], values);
    let sinks_2d = import_f32(&mut k, vec![1, hq as u32], &vec![5.0f32; hq]);
    let sinks = k.reshape(&sinks_2d, &[hq as u32]).unwrap();
    let bare = k
        .context
        .fused_sdpa(&q_t, &k_t, &v_t, 1.0 / 8.0, "", None, None)
        .unwrap();
    let sunk = k
        .context
        .fused_sdpa(&q_t, &k_t, &v_t, 1.0 / 8.0, "", None, Some(&sinks))
        .unwrap();
    let (bare, sunk) = (readback_f32(&mut k, &bare), readback_f32(&mut k, &sunk));
    let moved: f32 = bare.iter().zip(&sunk).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        moved > 0.05,
        "the sink vanished on the long GQA path (bare==sunk): {moved}"
    );
}

/// The 513 EDGE: at W = 513 a query at p sees [p-512, p] - the key at exactly
/// p-512 is VISIBLE, the one at p-513 is MASKED - checked against the oracle.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn the_window_edge_is_exactly_513_on_the_float_path() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let (hq, hkv, t, d, s) = (2, 1, 1, 64, 520);
    let p0 = (s - t) as u32; // 519; keys 0..519, origin 0
    let mut seed = 0x513u64;
    let q: Vec<f32> = (0..hq * t * d).map(|_| lcg(&mut seed)).collect();
    let kv: Vec<f32> = (0..2 * hkv * s * d).map(|_| lcg(&mut seed)).collect();
    let (keys, values) = kv.split_at(hkv * s * d);
    let q_t = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let k_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], keys);
    let v_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], values);
    let out = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: p0,
            key_origin: 0,
            window: Some(513),
            sinks: None,
            scale: 1.0 / (d as f32).sqrt(),
        })
        .unwrap();
    let got = readback_f32(&mut k, &out);
    let want = sdpa_f64(
        &q.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &keys.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &values.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        t,
        hq,
        s,
        d,
        hq / hkv,
        1.0 / (d as f64).sqrt(),
        p0,
        0,
        Some(513),
        None,
    );
    for (g, w) in got.iter().zip(&want) {
        assert!((g - *w as f32).abs() < 2.0e-2, "{g} vs {w}");
    }
}

/// PER QUERY HEAD sinks on the REAL float kernel, DETERMINISTIC closed form:
/// ZERO Q and K (every score exactly 0) and CONSTANT V, so without a sink every
/// head outputs mean(V); a raw sink s_h changes only the denominator
/// (S + exp(s_h)). Heads within a GQA group get DIFFERENT sinks -> different
/// outputs; a KV-head fold (one sink per group) makes them EQUAL -> fails.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn per_head_sinks_differ_on_the_float_path() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (4, 1, 1, 64, 16);
    let q = vec![0.0f32; hq * t * d];
    let keys = vec![0.0f32; hkv * s * d];
    let values = vec![1.0f32; hkv * s * d]; // constant V: mean is exactly 1
    let q_t = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let k_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &keys);
    let v_t = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &values);
    // heads (0,1) and (2,3) share the KV head; sinks differ WITHIN each pair
    let sinks_2d = import_f32(&mut k, vec![1, hq as u32], &[0.0, 4.0, 0.0, 4.0]);
    let out = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: s as u32 - 1,
            key_origin: 0,
            window: None,
            sinks: Some(&sinks_2d),
            scale: 1.0 / (d as f32).sqrt(),
        })
        .unwrap();
    let got = readback_f32(&mut k, &out);
    for (h, sink) in [(0usize, 0.0f64), (1, 4.0), (2, 0.0), (3, 4.0)] {
        let want = s as f64 / (s as f64 + (sink - 0.0).exp());
        let got_mean: f32 = got[h * d..(h + 1) * d].iter().sum::<f32>() / d as f32;
        assert!(
            (got_mean as f64 - want).abs() < 5.0e-2,
            "head {h}: closed form {want} vs {got_mean}"
        );
    }
    // the within-group heads differ (a fold collapses them)
    let mean = |h: usize| -> f32 { got[h * d..(h + 1) * d].iter().sum::<f32>() / d as f32 };
    assert!(
        (mean(0) - mean(1)).abs() > 0.05,
        "per-head sinks must not fold"
    );
    assert!(
        (mean(0) - mean(2)).abs() < 1.0e-3,
        "same sinks => same output"
    );
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
/// The BATCH BF16 matrix (the 16577 (b) contract): real shared_attention on BF16
/// tensors at D=128, Hq=32/Hkv=4 (GQA 8), a SHIFTED key origin, both full and
/// window-513 modes, T=2/7/17/512/1024 - T=2 rides the vendor fused SDPA (16 <= 32),
/// every larger T rides the FLASH float arm (the shape dispatch). Compared against
/// the f64 oracle fed the exact BF16-ROUNDED values WITH the layout transposes:
/// the native tensors are head-major [1, H, T, D], the oracle is row-major
/// [T, H, D], and the output comparison maps the indices - the axis debt #4
/// caught (a [T, H, D] slice of a head-major buffer reads 3.8125 where 1 is true).
///
/// Mixed-sign inputs use the coordinator's row-scaled 2-ULP gate for both paths.
/// Every element is checked against the reference row's maximum magnitude.
/// Detailed per-path error analysis is a follow-up, not a store-only claim.
fn bf16_batch_rows_match_the_rounded_oracle() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, d, s) = (32, 4, 128, 1100);
    let origin = 3u32; // shifted, not zero
    let mut seed = 0xB16B00B5u64;
    let q32: Vec<f32> = (0..hq * 1024 * d).map(|_| lcg(&mut seed)).collect();
    let kv32: Vec<f32> = (0..2 * hkv * s * d).map(|_| lcg(&mut seed)).collect();
    let (k32, v32) = kv32.split_at(hkv * s * d);
    // native [1, H, T, D] -> oracle row-major [T, H, D]
    let to_row_major = |native: &[f64], h: usize, t: usize, d: usize| -> Vec<f64> {
        let mut row = vec![0.0f64; t * h * d];
        for tt in 0..t {
            for hh in 0..h {
                for c in 0..d {
                    row[(tt * h + hh) * d + c] = native[(hh * t + tt) * d + c];
                }
            }
        }
        row
    };
    // the FULL T x mode cross (the earlier 17/512/1024 were window-only - #4
    // review) plus the T4/T5 boundary cells around GQA8*T = 32: T4 rides the
    // vendor arm, T5 the flash arm, the same oracle checks both sides of the
    // dispatch boundary.
    for (case, (t, window)) in [
        (2usize, None),
        (2, Some(513u32)),
        (4, None), // boundary: 4*8 = 32, the vendor arm's exact limit
        (4, Some(513)),
        (5, None), // boundary: 5*8 = 40, the first FLASH-arm row count
        (5, Some(513)),
        (7, None),
        (7, Some(513)),
        (17, None),
        (17, Some(513)),
        (512, None),
        (512, Some(513)),
        (1024, None),
        (1024, Some(513)),
    ]
    .into_iter()
    .enumerate()
    {
        let q_src = import_f32(
            &mut k,
            vec![1, hq as u32, t as u32, d as u32],
            &q32[..hq * t * d],
        );
        let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
        let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], k32);
        let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
        let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], v32);
        let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
        let to_f64 = |data: Vec<f32>| -> Vec<f64> { data.into_iter().map(|v| v as f64).collect() };
        // the oracle sees the ROUNDED values in ITS row-major layout
        let qr = to_row_major(&to_f64(readback_f32(&mut k, &q_t)), hq, t, d);
        let kr = to_row_major(&to_f64(readback_f32(&mut k, &k_t)), hkv, s, d);
        let vr = to_row_major(&to_f64(readback_f32(&mut k, &v_t)), hkv, s, d);
        let p0 = origin + s as u32 - t as u32;
        let out = k
            .shared_attention(SharedAttention {
                q: &q_t,
                kv: KvBlock::Float {
                    keys: &k_t,
                    values: &v_t,
                },
                query_start: p0,
                key_origin: origin,
                window,
                sinks: None,
                scale: 1.0 / (d as f32).sqrt(),
            })
            .unwrap();
        let got = readback_f32(&mut k, &out);
        let want = sdpa_f64(
            &qr,
            &kr,
            &vr,
            t,
            hq,
            s,
            d,
            hq / hkv,
            1.0 / (d as f64).sqrt(),
            p0,
            origin,
            window,
            None,
        );
        let path = if t * (hq / hkv) <= 32 {
            "vendor"
        } else {
            "flash"
        };
        let (worst, offender) = pinned_max_ratio(&got, &want, hq, t, d);
        println!(
            "case {case} T={t} W={window:?} {path} maxratio={worst:.2} h={} t={} c={} ref={:.6} got={:.6} rowmax={:.6} scale={:.6} ulp={:.8}",
            offender.0,
            offender.1,
            offender.2,
            offender.4,
            offender.3,
            offender.5,
            offender.6,
            ulp_bf16(offender.6)
        );
        assert!(
            worst <= 2.0,
            "T={t} window={window:?} path={path}: maxratio {worst:.2} exceeds the PINNED 2-ULP gate"
        );
    }
}

/// #4's COMPANION cell (the P-narrowing guard): GQA 8, T=7 (forced flash), zero Q,
/// K = BF16 2^17, V = BF16 2^30, raw sink 30*ln2. The analytic output per row is
/// EXACTLY its visible-key count (S_visible = 1..7): every score is 0, the sink
/// takes m to ~30 (log2), each key's probability is 2^-30 - which UNDERFLOWS half
/// to zero (O collapses to 0 if P stages as half) but survives F32 staging. Small
/// integers are exact in bf16, so the expectation is exact.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn flash_bf16_tiny_probabilities_survive_the_pt_staging() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (32, 4, 7, 128, 7);
    let q = vec![0.0f32; hq * t * d];
    let big_k = vec![(2.0f32).powi(17); hkv * s * d];
    let big_v = vec![(2.0f32).powi(30); hkv * s * d];
    let q_src = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
    let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &big_k);
    let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
    let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &big_v);
    let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
    // raw sink 30*ln2 per query head (30 in the log2 domain)
    let sink_val = 30.0f32 * std::f32::consts::LN_2;
    let sinks_2d = import_f32(&mut k, vec![1, hq as u32], &vec![sink_val; hq]);
    let out = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: s as u32 - t as u32, // p0 = 0, origin 0
            key_origin: 0,
            window: None,
            sinks: Some(&sinks_2d),
            scale: 1.0 / (d as f32).sqrt(),
        })
        .unwrap();
    let got = readback_f32(&mut k, &out);
    // NATIVE layout [1, Hq, T, D]: element (h, i, c) sits at (h * t + i) * d + c.
    // EVERY element, EXACT: small integers are exact in bf16, and a perfect kernel
    // must read (i + 1) for every (h, c) - no mean, no slack (#4's layout fix).
    for h in 0..hq {
        for i in 0..t {
            for c in 0..d {
                let idx = (h * t + i) * d + c;
                let want = (i + 1) as f32;
                assert!(
                    (got[idx] - want).abs() < 1.0e-3,
                    "head {h} row {i} ch {c}: analytic {want} vs {}",
                    got[idx]
                );
            }
        }
    }
}

/// #4's LOUD W513 boundary cell (the sharp geometry detector diffuse randoms
/// cannot be): zero Q and K (every score 0 -> uniform attention over exactly the
/// visible keys), GQA 8, T=7 on the FLASH arm, p0 = origin + 512, S = 519, and V
/// = 512 ONLY at union index 0 (0 elsewhere). W513: the key at origin is exactly
/// 512 behind the first query row - INSIDE the window - so each row's output is
/// V's one-hot mass: expected [1, 0, .., 0] per head. W512: that key falls OUT -
/// the output is exactly 0. A one-key window error moves the output by 1.0, not
/// 1/513.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn the_w513_boundary_is_loud_not_diffuse() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (32, 4, 7, 128, 519);
    let origin = 5u32;
    let q = vec![0.0f32; hq * t * d];
    let keys = vec![0.0f32; hkv * s * d];
    let mut values = vec![0.0f32; hkv * s * d];
    for h in 0..hkv {
        for c in 0..d {
            values[h * s * d + c] = 512.0; // union index 0 only
        }
    }
    let q_src = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
    let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &keys);
    let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
    let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &values);
    let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
    let p0 = origin + 512; // the first query row sits 512 ahead of union index 0
    for (window, expect_mass) in [(513u32, true), (512, false)] {
        let out = k
            .shared_attention(SharedAttention {
                q: &q_t,
                kv: KvBlock::Float {
                    keys: &k_t,
                    values: &v_t,
                },
                query_start: p0,
                key_origin: origin,
                window: Some(window),
                sinks: None,
                scale: 1.0 / (d as f32).sqrt(),
            })
            .unwrap();
        let got = readback_f32(&mut k, &out);
        // Only W513 row zero sees the boundary key; 512/513 rounds to BF16 1.0.
        // W512 and all later rows exclude it, so every channel is exactly zero.
        for h in 0..hq {
            for i in 0..t {
                let want = if expect_mass && i == 0 { 1.0 } else { 0.0 };
                for c in 0..d {
                    assert_eq!(
                        got[(h * t + i) * d + c],
                        want,
                        "W{window} h={h} row={i} ch={c}"
                    );
                }
            }
        }
        println!("W{window} boundary: expect_mass={expect_mass} VERIFIED");
    }
}

/// PER-QUERY-HEAD sinks ON THE FLASH ARM (the adoption bar's cell): GQA 8, T=7,
/// zero Q/K, V = BF16 2^30, sink_h = (30 + h%4)*ln2 per head. Expected EXACT:
/// out[h][i][c] = (i+1) * 2^(-(h%4)) - row i sees i+1 keys, each key's mass
/// 2^-(30+h%4) against the sink's 1, times V = 2^30 leaves (i+1)/2^(h%4),
/// exact in bf16 (small integers scaled by binary powers).
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn per_head_sinks_on_the_flash_arm() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (32, 4, 7, 128, 7);
    let q = vec![0.0f32; hq * t * d];
    let keys = vec![0.0f32; hkv * s * d];
    let big_v = vec![(2.0f32).powi(30); hkv * s * d];
    let q_src = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
    let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &keys);
    let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
    let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &big_v);
    let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
    // sink_h = (30 + h%4)*ln2 - the raw logit whose log2 value is 30 + h%4
    let sinks: Vec<f32> = (0..hq)
        .map(|h| (30.0 + (h % 4) as f32) * std::f32::consts::LN_2)
        .collect();
    let sinks_2d = import_f32(&mut k, vec![1, hq as u32], &sinks);
    let out = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: s as u32 - t as u32,
            key_origin: 0,
            window: None,
            sinks: Some(&sinks_2d),
            scale: 1.0 / (d as f32).sqrt(),
        })
        .unwrap();
    let got = readback_f32(&mut k, &out);
    for h in 0..hq {
        let scale = 2f32.powi(-((h % 4) as i32));
        for i in 0..t {
            for c in 0..d {
                let want = (i + 1) as f32 * scale;
                let idx = (h * t + i) * d + c;
                // EXACT per the ruling: (i+1)*2^(-(h%4)) is dyadic - exactly
                // representable in bf16 - so every element is assert_eq.
                assert_eq!(
                    got[idx], want,
                    "h={h} i={i} c={c}: dyadic expectation must be EXACT"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn small_f16_float_preserves_mask_and_raw_sinks() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let [q, keys, values] = [
        (vec![1, 8, 1, 64], vec![0u8; 8 * 64 * 2]),
        (vec![1, 1, 16, 64], vec![0u8; 16 * 64 * 2]),
        (
            vec![1, 1, 16, 64],
            half::f16::MAX.to_bits().to_le_bytes().repeat(16 * 64),
        ),
    ]
    .map(|(shape, bytes)| {
        let host = Tensor::try_from_host(Dtype::F16, shape, bytes).unwrap();
        k.import_tensor(&host).unwrap()
    });
    let raw = 20.006f32;
    let sinks = import_f32(&mut k, vec![1, 8], &[raw; 8]);
    let reference = (65504.0 * 16.0 / (16.0 + f64::from(raw).exp())) as f32;
    let rounded_raw = f64::from(half::f16::from_f32(raw).to_f32());
    let rounded_reference = (65504.0 * 16.0 / (16.0 + rounded_raw.exp())) as f32;
    assert_ne!(
        half::f16::from_f32(reference),
        half::f16::from_f32(rounded_reference)
    );
    for with_sink in [false, true] {
        let out = k
            .shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: 15,
                key_origin: 0,
                window: None,
                sinks: with_sink.then_some(&sinks),
                scale: 0.125,
            })
            .unwrap();
        assert_eq!(out.shape(), q.shape());
        assert_eq!(out.dtype(), Dtype::F16);
        let host = k.readback(&out).unwrap();
        let bytes = host.host_bytes().unwrap();
        assert_eq!(bytes.len(), 8 * 64 * 2);
        let expected = half::f16::from_f32(if with_sink { reference } else { 65504.0 }).to_bits();
        for (index, bits) in bytes.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(
                u16::from_le_bytes(*bits),
                expected,
                "raw_sink={with_sink} index={index}"
            );
        }
        assert_eq!(readback_f32(&mut k, &sinks), vec![raw; 8]);
    }
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn shared_sinks_refuse_before_import_and_leave_owner_usable() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let q = import_f32(&mut k, vec![1, 8, 1, 64], &[0.0; 8 * 64]);
    let keys = import_f32(&mut k, vec![1, 1, 16, 64], &[0.0; 16 * 64]);
    let values = import_f32(&mut k, vec![1, 1, 16, 64], &[1.0; 16 * 64]);
    let sinks = import_f32(&mut k, vec![1, 8], &[0.0; 8]);
    let flat = k.reshape(&sinks, &[8]).unwrap();
    let column = k.reshape(&sinks, &[8, 1]).unwrap();
    let bf16 = k.cast(&sinks, Dtype::Bf16).unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let foreign = import_f32(&mut other, vec![1, 8], &[0.0; 8]);
    let request = || SharedAttention {
        q: &q,
        kv: KvBlock::Float {
            keys: &keys,
            values: &values,
        },
        query_start: 15,
        key_origin: 0,
        window: None,
        sinks: None,
        scale: 0.125,
    };
    // Armed, BF16 is the SUPPORTED sink storage (the flagged model stores
    // them that way), so the BF16 refusal is the UNFLAGGED contract; the
    // flat/column/foreign shapes must refuse under either flag.
    let flagged = crate::kernels::mlx::decode_attn_flagged();
    let bad_sinks: Vec<&ExecutionTensor> = if flagged {
        vec![&flat, &column, &foreign]
    } else {
        vec![&flat, &column, &bf16, &foreign]
    };
    for bad in bad_sinks {
        #[cfg(feature = "native-fault-qualification")]
        let syncs = k.qualification_sync_count();
        let error = k
            .shared_attention(SharedAttention {
                sinks: Some(bad),
                ..request()
            })
            .err()
            .expect("malformed shared sink must refuse");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        #[cfg(feature = "native-fault-qualification")]
        assert_eq!(
            k.qualification_sync_count(),
            syncs,
            "bad sinks must refuse before mask imports"
        );
        let out = k.shared_attention(request()).unwrap();
        assert_eq!(readback_f32(&mut k, &out), vec![1.0; 8 * 64]);
    }
}

/// First greedy-step shapes after a 2K prefix: raw F32 sinks must not be
/// rounded to BF16 just to satisfy the vendor SDPA dtype check. Since the
/// sink-exactness gate: unarmed, a raw-F32 publisher keeps the flash route
/// (exact F32 handling); armed, these non-BF16-exact ln2-multiple sinks
/// make the vendor promotion REFUSE loudly instead of rounding.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn small_bf16_float_keeps_raw_f32_sinks() {
    let _serial = crate::native_test_lock();
    let flagged = crate::kernels::mlx::decode_attn_flagged();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    for s in [513u32, 2049] {
        let q_src = import_f32(&mut k, vec![1, 32, 1, 128], &vec![0.0; 32 * 128]);
        let q = k.cast(&q_src, Dtype::Bf16).unwrap();
        let key_src = import_f32(&mut k, vec![1, 4, s, 128], &vec![0.0; 4 * s as usize * 128]);
        let keys = k.cast(&key_src, Dtype::Bf16).unwrap();
        let value_src = import_f32(
            &mut k,
            vec![1, 4, s, 128],
            &vec![2.0f32.powi(30); 4 * s as usize * 128],
        );
        let values = k.cast(&value_src, Dtype::Bf16).unwrap();
        let raw: Vec<f32> = (0..32)
            .map(|h| (30.0 + (h % 4) as f32) * std::f32::consts::LN_2)
            .collect();
        let sinks = import_f32(&mut k, vec![1, 32], &raw);
        let out = k.shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 2048,
            key_origin: 2049 - s,
            window: (s == 513).then_some(513),
            sinks: Some(&sinks),
            scale: 1.0 / 128.0f32.sqrt(),
        });
        if flagged {
            // These ln2-multiple sinks are NOT BF16-exact: the vendor
            // promotion must refuse (BadRequest naming the exactness
            // failure), never round them silently.
            // .err().expect - Result::expect_err needs Debug on the Ok type.
            let err = out
                .err()
                .expect("non-exact sinks must refuse the vendor arm");
            assert!(
                matches!(err, EngError::BadRequest(_)),
                "refusal must be BadRequest: {err:?}"
            );
            assert!(format!("{err:?}").contains("BF16"), "diagnostic: {err:?}");
            assert_eq!(readback_f32(&mut k, &sinks), raw);
            continue;
        }
        let out = out.unwrap();
        assert_eq!(out.shape(), &[1, 32, 1, 128]);
        assert_eq!(out.dtype(), Dtype::Bf16);
        let got = readback_f32(&mut k, &out);
        assert_eq!(got.len(), 32 * 128);
        // For S=2^n+1, the sink-dominated BF16 result rounds to (S-1)*2^(-h%4).
        for (h, row) in got.as_chunks::<128>().0.iter().enumerate() {
            let want = (s - 1) as f32 * 2.0f32.powi(-((h % 4) as i32));
            for (c, &value) in row.iter().enumerate() {
                assert_eq!(value, want, "S={s} head={h} channel={c}");
            }
        }
        assert_eq!(readback_f32(&mut k, &sinks), raw);
    }
}

/// Flagged-path sink-dtype gate: with FIFI_FLINT_MLX_DECODE_ATTN=1 the
/// vendor decode arm takes sinks in the OUTPUT dtype. BF16-stored sinks
/// (the flagged model's storage) pass through; a raw-F32 publisher is
/// PROMOTED only when every value is BF16-exact (API-side readback) and
/// NON-exact values REFUSE with BadRequest. Pinned through the real
/// dispatch with the route counters; the no-sink arm proves the sink
/// mass still moves the output. Armed by default since the 2026-10-27
/// flip (a =0 process skips this).
#[test]
#[ignore = "requires an explicit Mac GPU allocation; armed by default, =0 skips it"]
fn flagged_sink_dtype_gates_decode_vendor() {
    if !crate::kernels::mlx::decode_attn_flagged() {
        return;
    }
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    // Integer sinks are BF16-exact by construction (mantissa fits); the
    // values sit in the checkpoint's real range (4.0..11.0).
    let raw: Vec<f32> = (0..32).map(|h| 4.0 + (h % 8) as f32).collect();
    for &value in &raw {
        assert_eq!(
            crate::kernels::mlx::bf16_round_trip(value).to_bits(),
            value.to_bits(),
            "fixture sinks must be BF16-exact"
        );
    }
    let sinks = import_f32(&mut k, vec![1, 32], &raw);
    let sinks_bf16 = k.cast(&sinks, Dtype::Bf16).unwrap();
    let s = 513u32;
    let q_src = import_f32(&mut k, vec![1, 32, 1, 128], &vec![0.0; 32 * 128]);
    let q = k.cast(&q_src, Dtype::Bf16).unwrap();
    let key_src = import_f32(&mut k, vec![1, 4, s, 128], &vec![0.0; 4 * s as usize * 128]);
    let keys = k.cast(&key_src, Dtype::Bf16).unwrap();
    let value_src = import_f32(
        &mut k,
        vec![1, 4, s, 128],
        &vec![2.0f32.powi(20); 4 * s as usize * 128],
    );
    let values = k.cast(&value_src, Dtype::Bf16).unwrap();
    // Imported BEFORE the run closure: the closure holds &mut k for the
    // rest of the test, so a later import would be E0499.
    let mut inexact: Vec<f32> = raw.clone();
    inexact[0] += 1.0 / (1 << 20) as f32;
    let sinks_inexact = import_f32(&mut k, vec![1, 32], &inexact);
    let mut run = |sinks: Option<&ExecutionTensor>| -> Result<Vec<u8>, EngError> {
        let out = k.shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 512,
            key_origin: 513 - s,
            window: Some(513),
            sinks,
            scale: 1.0 / 128.0f32.sqrt(),
        })?;
        let f = k.cast(&out, Dtype::F32)?;
        let host = k.readback(&f)?;
        let bytes = host
            .host_bytes()
            .ok_or_else(|| EngError::BadRequest("sink readback empty".into()))?;
        Ok(bytes.to_vec())
    };
    // BF16-stored sinks (the loader's storage): the vendor arm, no flash.
    crate::kernels::mlx::native::attn_dispatch::reset();
    let with_bf16 = run(Some(&sinks_bf16)).expect("BF16 sinks dispatch cleanly");
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(vendor, 1, "BF16-stored sinks must ride the vendor arm");
    assert_eq!(
        flash, 0,
        "the BF16-sink decode shape must not fall back to flash"
    );
    // Exact raw F32 sinks: the API-side check promotes and the vendor arm
    // still serves the shape (float flash is REP-8; checked promotion is
    // the only exact route for displaced GQA geometries) - and the promoted
    // output must match the stored-BF16 output bitwise (same values, same
    // kernel).
    crate::kernels::mlx::native::attn_dispatch::reset();
    let with_f32 = run(Some(&sinks)).expect("BF16-exact F32 sinks promote cleanly");
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(vendor, 1, "BF16-exact F32 sinks must ride the vendor arm");
    assert_eq!(
        flash, 0,
        "exactness-checked promotion must not detour to flash"
    );
    assert_eq!(
        with_f32, with_bf16,
        "promoted F32 sinks must match stored BF16 sinks bitwise"
    );
    // NON-representable F32 sinks: loud refusal, nothing dispatched.
    crate::kernels::mlx::native::attn_dispatch::reset();
    let refused = run(Some(&sinks_inexact));
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(
        (flash, vendor),
        (0, 0),
        "a refused call must dispatch nothing"
    );
    let err = refused.expect_err("non-BF16-exact F32 sinks must refuse");
    assert!(
        matches!(err, EngError::BadRequest(_)),
        "refusal must be BadRequest: {err:?}"
    );
    assert!(format!("{err:?}").contains("BF16"), "diagnostic: {err:?}");
    // The sink mass still moves the vendor output vs no sinks.
    let bare = run(None).expect("the no-sink shape dispatches cleanly");
    assert_ne!(
        with_bf16, bare,
        "dropping the sink must change the armed output"
    );
}

/// GQA4 promotion pin (flagged route): float flash refuses hq/hkv != 8
/// (REP-8), so a one-row BF16 call at GQA 4 with raw F32 sinks has NO
/// flash route - the API-side exactness-checked promotion is what keeps
/// this geometry executable. Exact values ride the vendor arm and match
/// stored-BF16 sinks bitwise; non-exact values refuse with zero
/// dispatches. Runs armed (the default since the 2026-10-27 flip).
#[test]
#[ignore = "requires an explicit Mac GPU allocation; armed by default, =0 skips it"]
fn gqa4_sink_promotion_keeps_vendor_route() {
    if !crate::kernels::mlx::decode_attn_flagged() {
        return;
    }
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    // 16 query heads over 4 KV heads = GQA 4; integer sinks are BF16-exact.
    let raw: Vec<f32> = (0..16).map(|h| 4.0 + (h % 8) as f32).collect();
    for &value in &raw {
        assert_eq!(
            crate::kernels::mlx::bf16_round_trip(value).to_bits(),
            value.to_bits(),
            "fixture sinks must be BF16-exact"
        );
    }
    let sinks = import_f32(&mut k, vec![1, 16], &raw);
    let sinks_bf16 = k.cast(&sinks, Dtype::Bf16).unwrap();
    let s = 513u32;
    let q_src = import_f32(&mut k, vec![1, 16, 1, 128], &vec![0.0; 16 * 128]);
    let q = k.cast(&q_src, Dtype::Bf16).unwrap();
    let key_src = import_f32(&mut k, vec![1, 4, s, 128], &vec![0.0; 4 * s as usize * 128]);
    let keys = k.cast(&key_src, Dtype::Bf16).unwrap();
    let value_src = import_f32(
        &mut k,
        vec![1, 4, s, 128],
        &vec![2.0f32.powi(20); 4 * s as usize * 128],
    );
    let values = k.cast(&value_src, Dtype::Bf16).unwrap();
    let mut inexact = raw.clone();
    inexact[0] += 1.0 / (1 << 20) as f32;
    let sinks_inexact = import_f32(&mut k, vec![1, 16], &inexact);
    let mut run = |sinks: Option<&ExecutionTensor>| -> Result<Vec<u8>, EngError> {
        let out = k.shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 512,
            key_origin: 513 - s,
            window: Some(513),
            sinks,
            scale: 1.0 / 128.0f32.sqrt(),
        })?;
        let f = k.cast(&out, Dtype::F32)?;
        let host = k.readback(&f)?;
        let bytes = host
            .host_bytes()
            .ok_or_else(|| EngError::BadRequest("sink readback empty".into()))?;
        Ok(bytes.to_vec())
    };
    crate::kernels::mlx::native::attn_dispatch::reset();
    let with_bf16 = run(Some(&sinks_bf16)).expect("BF16 sinks dispatch cleanly");
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(vendor, 1, "GQA4 BF16-stored sinks must ride the vendor arm");
    assert_eq!(flash, 0, "GQA4 has no flash route; vendor is the only arm");
    crate::kernels::mlx::native::attn_dispatch::reset();
    let with_f32 = run(Some(&sinks)).expect("exact F32 sinks promote cleanly");
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(vendor, 1, "GQA4 exact F32 sinks must ride the vendor arm");
    assert_eq!(flash, 0);
    assert_eq!(
        with_f32, with_bf16,
        "promoted F32 sinks must match stored BF16 sinks bitwise at GQA4"
    );
    crate::kernels::mlx::native::attn_dispatch::reset();
    let refused = run(Some(&sinks_inexact));
    let (flash, vendor) = crate::kernels::mlx::native::attn_dispatch::counts();
    assert_eq!(
        (flash, vendor),
        (0, 0),
        "a refused GQA4 call must dispatch nothing"
    );
    let err = refused.expect_err("non-BF16-exact GQA4 sinks must refuse");
    assert!(
        matches!(err, EngError::BadRequest(_)),
        "refusal must be BadRequest: {err:?}"
    );
    // The BadRequest must carry the helper's diagnostic VERBATIM: the
    // provenance, the first offending head and value, and the =0 remedy.
    let text = format!("{err:?}");
    assert!(
        text.contains("shared_attention decode vendor"),
        "provenance: {text}"
    );
    assert!(text.contains("head 0"), "first offending head: {text}");
    assert!(
        text.contains(&format!("value {:e}", inexact[0])),
        "offending value: {text}"
    );
    assert!(
        text.contains("FIFI_FLINT_MLX_DECODE_ATTN=0"),
        "rollback remedy: {text}"
    );
}

/// The FLASH arm's BF16-sink storage up-cast (flagged route): multi-row
/// flash calls and the capacity-plane fallback both take the flash kernel,
/// which up-casts BF16-stored sinks to F32 exactly. BF16-REPRESENTABLE
/// published values must produce BITWISE identical outputs either way -
/// storage must not perturb the flash arm's sink mass. Runs armed (the
/// default since the 2026-10-27 flip); the unflagged API refuses BF16
/// sinks outright (pinned elsewhere).
#[test]
#[ignore = "requires an explicit Mac GPU allocation; runs armed by default, =0 skips it"]
fn flash_route_bf16_sink_storage_upcasts_exactly() {
    if !crate::kernels::mlx::decode_attn_flagged() {
        return;
    }
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let raw: Vec<f32> = (0..32u32).map(|h| 4.0 + (h % 8) as f32).collect();
    for &value in &raw {
        assert_eq!(
            crate::kernels::mlx::bf16_round_trip(value).to_bits(),
            value.to_bits(),
            "fixture sinks must be BF16-exact"
        );
    }
    let sinks = import_f32(&mut k, vec![1, 32], &raw);
    let sinks_bf16 = k.cast(&sinks, Dtype::Bf16).unwrap();
    // Multi-row flash tensors: rows=64 planes, queries at 0..64.
    let s = 64u32;
    let q = import_f32(
        &mut k,
        vec![1, 32, s, 128],
        &vec![0.25; 32 * s as usize * 128],
    );
    let q = k.cast(&q, Dtype::Bf16).unwrap();
    let keys = import_f32(&mut k, vec![1, 4, s, 128], &vec![0.5; 4 * s as usize * 128]);
    let keys = k.cast(&keys, Dtype::Bf16).unwrap();
    let values = import_f32(
        &mut k,
        vec![1, 4, s, 128],
        &vec![2.0f32.powi(20); 4 * s as usize * 128],
    );
    let values = k.cast(&values, Dtype::Bf16).unwrap();
    // Capacity-plane fallback tensors: T=1 with s beyond the exact union
    // takes the flash kernel even armed (the capacity-plane guard).
    let cap = 1024u32;
    let len = 64u32;
    let q1 = import_f32(&mut k, vec![1, 32, 1, 128], &vec![0.25; 32 * 128]);
    let q1 = k.cast(&q1, Dtype::Bf16).unwrap();
    let keys_cap = import_f32(
        &mut k,
        vec![1, 4, cap, 128],
        &vec![0.5; 4 * cap as usize * 128],
    );
    let keys_cap = k.cast(&keys_cap, Dtype::Bf16).unwrap();
    let values_cap = import_f32(
        &mut k,
        vec![1, 4, cap, 128],
        &vec![2.0f32.powi(20); 4 * cap as usize * 128],
    );
    let values_cap = k.cast(&values_cap, Dtype::Bf16).unwrap();
    // A free fn (no closure capture) so the kernel stays borrowable between
    // calls while tensors live across the whole comparison.
    fn run_flash(
        k: &mut MlxKernels,
        q: &ExecutionTensor,
        keys: &ExecutionTensor,
        values: &ExecutionTensor,
        query_start: u32,
        key_origin: u32,
        sinks: Option<&ExecutionTensor>,
    ) -> Vec<u8> {
        let out = k
            .shared_attention(SharedAttention {
                q,
                kv: KvBlock::Float { keys, values },
                query_start,
                key_origin,
                window: None,
                sinks,
                scale: 1.0f32 / 128.0f32.sqrt(),
            })
            .unwrap();
        let f = k.cast(&out, Dtype::F32).unwrap();
        k.readback(&f).unwrap().host_bytes().unwrap().to_vec()
    }
    let f32_out = run_flash(&mut k, &q, &keys, &values, 0, 0, Some(&sinks));
    let bf16_out = run_flash(&mut k, &q, &keys, &values, 0, 0, Some(&sinks_bf16));
    assert_eq!(
        f32_out, bf16_out,
        "multi-row flash: BF16-stored sinks must match F32-published bitwise"
    );
    let f32_cap = run_flash(
        &mut k,
        &q1,
        &keys_cap,
        &values_cap,
        len - 1,
        0,
        Some(&sinks),
    );
    let bf16_cap = run_flash(
        &mut k,
        &q1,
        &keys_cap,
        &values_cap,
        len - 1,
        0,
        Some(&sinks_bf16),
    );
    assert_eq!(
        f32_cap, bf16_cap,
        "capacity-plane flash: BF16-stored sinks must match F32-published bitwise"
    );
    let bare = run_flash(&mut k, &q1, &keys_cap, &values_cap, len - 1, 0, None);
    assert_ne!(
        f32_cap, bare,
        "dropping the sink must change the flash output"
    );
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn large_f32_d256_respects_device_threadgroup_memory() {
    let _serial = crate::native_test_lock();
    assert_eq!(native::float_flash_staging_bytes(Dtype::F32, 256), 33_792);
    assert_eq!(native::float_flash_staging_bytes(Dtype::Bf16, 128), 13_056);
    assert_eq!(native::float_flash_staging_bytes(Dtype::Bf16, 256), 25_344);
    assert_eq!(native::float_flash_staging_bytes(Dtype::F16, 256), 25_344);
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let q = import_f32(&mut k, vec![1, 8, 5, 256], &vec![0.0; 8 * 5 * 256]);
    let keys = import_f32(&mut k, vec![1, 1, 5, 256], &vec![0.0; 5 * 256]);
    let values = import_f32(&mut k, vec![1, 1, 5, 256], &vec![2.0f32.powi(30); 5 * 256]);
    let raw: Vec<f32> = (0..8)
        .map(|h| (30.0 + (h % 4) as f32) * std::f32::consts::LN_2)
        .collect();
    let sinks = import_f32(&mut k, vec![1, 8], &raw);
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let q = k.cast(&q, dtype).unwrap();
        let keys = k.cast(&keys, dtype).unwrap();
        let values = k.cast(&values, dtype).unwrap();
        if !k.context.float_flash_fits(dtype, 256).unwrap() {
            let err = k
                .context
                .flash_prefill_float(&q, &keys, &values, 5, 1.0 / 16.0, &sinks, 0)
                .err()
                .expect("over-cap flash must refuse before native dispatch");
            assert!(
                matches!(&err, EngError::BadRequest(message) if message.contains("threadgroup-memory")),
                "{err:?}"
            );
        }
        let out = k
            .shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: 0,
                key_origin: 0,
                window: None,
                sinks: Some(&sinks),
                scale: 1.0 / 16.0,
            })
            .unwrap();
        assert_eq!(out.shape(), &[1, 8, 5, 256]);
        assert_eq!(out.dtype(), dtype);
        let got = readback_f32(&mut k, &out);
        assert_eq!(got.len(), 8 * 5 * 256);
        for (h, head) in got.as_chunks::<{ 5 * 256 }>().0.iter().enumerate() {
            for (i, row) in head.as_chunks::<256>().0.iter().enumerate() {
                let n = (i + 1) as f64;
                let want = n * 2.0f64.powi(30) / (n + f64::from(raw[h]).exp());
                for &value in row {
                    assert!((f64::from(value) - want).abs() <= 2.0 * ulp_bf16(want.abs()));
                }
            }
        }
    }
    assert_eq!(readback_f32(&mut k, &sinks), raw);
}

/// The SHORT-PREFIX regression (the first-block rule): T=512, S=512, p0=origin=0,
/// W513 - the window exceeds the prefix, which must RETAIN ALL keys (the safety
/// fix; this used to be a BadRequest). Zero Q/K, V=1 everywhere: row i's output
/// is the uniform mean over its i+1 visible keys = exactly 1.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn the_first_block_short_prefix_w513_is_valid() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (32, 4, 512, 128, 512);
    let q = vec![0.0f32; hq * t * d];
    let keys = vec![0.0f32; hkv * s * d];
    let ones = vec![1.0f32; hkv * s * d];
    let q_src = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
    let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &keys);
    let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
    let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], &ones);
    let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
    let out = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: 0,
            key_origin: 0,
            window: Some(513), // wider than the 512-key prefix: retain all
            sinks: None,
            scale: 1.0 / (d as f32).sqrt(),
        })
        .unwrap();
    let got = readback_f32(&mut k, &out);
    // EXACT per the ruling: the uniform mean of all-ones V is exactly 1.0 in
    // bf16 - every element, every channel, assert_eq.
    for h in 0..hq {
        for i in 0..t {
            for c in 0..d {
                let idx = (h * t + i) * d + c;
                assert_eq!(
                    got[idx], 1.0,
                    "h={h} i={i} c={c}: the uniform mean must be EXACTLY 1"
                );
            }
        }
    }
}

/// THE PREFLIGHT PROOFS (the agreed owner-usability debt): every typed
/// rejection returns BadRequest (never a panic, never a poisoning Backend
/// error), and a VALID operation still works on the same adapter right after
/// each rejection - the resident owner stays usable.
#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn rejections_are_bad_requests_and_the_owner_stays_usable() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let (hq, hkv, t, d, s) = (32, 4, 5, 128, 32); // Also contains the GQA probe's [14, 31) span.
    let mut seed = 0xFEEDu64;
    let q: Vec<f32> = (0..hq * t * d).map(|_| lcg(&mut seed)).collect();
    let kv: Vec<f32> = (0..2 * hkv * s * d).map(|_| lcg(&mut seed)).collect();
    let (keys, values) = kv.split_at(hkv * s * d);
    let q_src = import_f32(&mut k, vec![1, hq as u32, t as u32, d as u32], &q);
    let q_t = k.cast(&q_src, Dtype::Bf16).unwrap();
    let k_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], keys);
    let k_t = k.cast(&k_src, Dtype::Bf16).unwrap();
    let v_src = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], values);
    let v_t = k.cast(&v_src, Dtype::Bf16).unwrap();
    let valid = |k: &mut MlxKernels| {
        let result = k.shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: s as u32 - t as u32,
            key_origin: 0,
            window: None,
            sinks: None,
            scale: 1.0 / (d as f32).sqrt(),
        });
        match result {
            Ok(out) => readback_f32(k, &out).iter().all(|value| value.is_finite()),
            Err(_) => false,
        }
    };
    // a valid op works before any rejection
    assert!(valid(&mut k), "the valid op must work at the start");

    // (a) GQA != 8 refused on the FLASH arm; owner usable after. NOTE: the
    // refusal applies to the flash path (rows x GQA > 32) - a small-T request
    // legitimately rides the vendor arm at any GQA, so the probe must be big
    // enough to dispatch to flash: Hq=8/Hkv=4 (GQA 2) at T=17 -> 34 > 32.
    let bad_gqa_src = import_f32(&mut k, vec![1, 8, 17, 128], &vec![0.5f32; 8 * 17 * 128]);
    let bad_gqa_q = k.cast(&bad_gqa_src, Dtype::Bf16).unwrap();
    let refused = k
        .shared_attention(SharedAttention {
            q: &bad_gqa_q,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: 14,
            key_origin: 0,
            window: None,
            sinks: None,
            scale: 0.1,
        })
        .err()
        .expect("GQA-2 at flash-dispatch size must refuse");
    assert!(
        matches!(&refused, EngError::BadRequest(m) if m.contains("GQA == 8")),
        "{refused:?}"
    );
    assert!(valid(&mut k), "owner usable after the GQA rejection");

    // (b) dtype mismatch refused; owner usable after
    let f32_k = import_f32(&mut k, vec![1, hkv as u32, s as u32, d as u32], keys);
    let refused = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &f32_k,
                values: &v_t,
            },
            query_start: s as u32 - t as u32,
            key_origin: 0,
            window: None,
            sinks: None,
            scale: 0.1,
        })
        .err()
        .expect("the dtype mismatch must refuse");
    assert!(
        matches!(&refused, EngError::BadRequest(m) if m.to_ascii_lowercase().contains("matching")),
        "{refused:?}"
    );
    assert!(valid(&mut k), "owner usable after the dtype rejection");

    // (c) the degenerate Some(0) window refused; owner usable after
    let refused = k
        .shared_attention(SharedAttention {
            q: &q_t,
            kv: KvBlock::Float {
                keys: &k_t,
                values: &v_t,
            },
            query_start: s as u32 - t as u32,
            key_origin: 0,
            window: Some(0),
            sinks: None,
            scale: 0.1,
        })
        .err()
        .expect("Some(0) must refuse");
    assert!(
        matches!(&refused, EngError::BadRequest(m) if m.contains("0 masks")),
        "{refused:?}"
    );
    assert!(valid(&mut k), "owner usable after the window rejection");
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn small_float_rejects_bad_metadata_without_poisoning() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let q = import_f32(&mut k, vec![1, 8, 1, 64], &[0.0; 8 * 64]);
    let keys = import_f32(&mut k, vec![1, 1, 16, 64], &[0.0; 16 * 64]);
    let values = import_f32(&mut k, vec![1, 1, 16, 64], &[1.0; 16 * 64]);
    let batch_q = import_f32(&mut k, vec![2, 8, 1, 64], &[0.0; 2 * 8 * 64]);
    let batch_k = import_f32(&mut k, vec![2, 1, 16, 64], &[0.0; 2 * 16 * 64]);
    let batch_v = import_f32(&mut k, vec![2, 1, 16, 64], &[1.0; 2 * 16 * 64]);
    let flat_q = k.reshape(&q, &[8 * 64]).unwrap();
    let flat_k = k.reshape(&keys, &[16 * 64]).unwrap();
    let bf_q = k.cast(&q, Dtype::Bf16).unwrap();
    let bf_k = k.cast(&keys, Dtype::Bf16).unwrap();
    let bf_v = k.cast(&values, Dtype::Bf16).unwrap();
    let int_source = import_f32(&mut k, vec![1, 8, 1, 256], &[0.0; 8 * 256]);
    let (int_q, _, _) = k.quantize_rows(&int_source, 64, 8).unwrap();
    let valid = |k: &mut MlxKernels| {
        let out = k
            .shared_attention(SharedAttention {
                q: &q,
                kv: KvBlock::Float {
                    keys: &keys,
                    values: &values,
                },
                query_start: 15,
                key_origin: 0,
                window: None,
                sinks: None,
                scale: 0.125,
            })
            .unwrap();
        assert_eq!(readback_f32(k, &out), vec![1.0; 8 * 64]);
    };
    for (bad_q, bad_k, bad_v) in [
        (&batch_q, &batch_k, &batch_v),
        (&batch_q, &keys, &values),
        (&q, &batch_k, &batch_v),
        (&flat_q, &keys, &values),
        (&q, &flat_k, &values),
        (&bf_q, &keys, &values),
        (&q, &bf_k, &values),
        (&q, &keys, &bf_v),
        (&int_q, &keys, &values),
    ] {
        let error = k
            .shared_attention(SharedAttention {
                q: bad_q,
                kv: KvBlock::Float {
                    keys: bad_k,
                    values: bad_v,
                },
                query_start: 15,
                key_origin: 0,
                window: None,
                sinks: None,
                scale: 0.125,
            })
            .err()
            .expect("malformed shared operands must refuse");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        valid(&mut k);
        let error = k
            .context
            .fused_sdpa(bad_q, bad_k, bad_v, 0.125, "", None, None)
            .err()
            .expect("malformed fused operands must refuse");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        valid(&mut k);
    }
    let f32_mask = import_f32(&mut k, vec![1, 16], &[0.0; 16]);
    let error = k
        .context
        .fused_sdpa(&bf_q, &bf_k, &bf_v, 0.125, "", Some(&f32_mask), None)
        .err()
        .expect("mask must promote to the output dtype");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    valid(&mut k);
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn default_q8_sinks_reject_bad_query_rank_without_panicking() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let q = import_f32(&mut k, vec![1, 8, 1, 64], &[0.0; 8 * 64]);
    let bad_q = k.reshape(&q, &[8 * 64]).unwrap();
    let keys = import_f32(&mut k, vec![1, 1, 16, 64], &[0.0; 16 * 64]);
    let values = import_f32(&mut k, vec![1, 1, 16, 64], &[1.0; 16 * 64]);
    let (kc, ks, kb) = k.quantize_rows(&keys, 64, 8).unwrap();
    let (vc, vs, vb) = k.quantize_rows(&values, 64, 8).unwrap();
    let keys = QuantizedKv {
        codes: &kc,
        scales: &ks,
        biases: &kb,
    };
    let values = QuantizedKv {
        codes: &vc,
        scales: &vs,
        biases: &vb,
    };
    let error = k
        .q8_flash_prefill(&bad_q, &keys, &values, 16, 0.125, 64, 8, None, 0)
        .err()
        .expect("rank-one query must refuse before default sinks");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    let out = k
        .q8_flash_prefill(&q, &keys, &values, 16, 0.125, 64, 8, None, 0)
        .unwrap();
    assert_eq!(readback_f32(&mut k, &out), vec![1.0; 8 * 64]);
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn fused_sdpa_rejects_bad_ranks_without_poisoning() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let q = import_f32(&mut k, vec![1, 8, 1, 64], &vec![0.0; 8 * 64]);
    let keys = import_f32(&mut k, vec![1, 1, 16, 64], &vec![0.0; 16 * 64]);
    let values = import_f32(&mut k, vec![1, 1, 16, 64], &vec![1.0; 16 * 64]);
    let bad_mask = import_f32(&mut k, vec![1], &[0.0]);
    let bad_sinks = import_f32(&mut k, vec![8, 1], &[0.0; 8]);
    for (mask, sinks) in [(Some(&bad_mask), None), (None, Some(&bad_sinks))] {
        let error = k
            .context
            .fused_sdpa(&q, &keys, &values, 0.125, "", mask, sinks)
            .err()
            .expect("invalid rank must refuse");
        assert!(matches!(error, EngError::BadRequest(_)));
        let out = k
            .context
            .fused_sdpa(&q, &keys, &values, 0.125, "", None, None)
            .unwrap();
        assert!(readback_f32(&mut k, &out).iter().all(|&value| value == 1.0));
    }
}

fn two_key_union_last_row(k: &mut MlxKernels, rows: u32, origin: u32) -> Vec<f32> {
    let t = rows as usize;
    let slots = (2048 - origin) as usize;
    let mut queries = vec![0.0; 32 * t * 128];
    for query in queries.as_chunks_mut::<128>().0 {
        query[0] = 1.0;
    }
    let mut keys = vec![0.0; 4 * slots * 128];
    let mut values = keys.clone();
    for head in 0..4 {
        for position in origin..2048 {
            let offset = (head * slots + (position - origin) as usize) * 128;
            keys[offset] = match position {
                1535 => 0.0,
                1536 => 5.65625,
                _ => -1448.0,
            };
            if position == 1535 {
                values[offset..offset + 128].fill(1.0);
            }
        }
    }
    let q = import_f32(k, vec![1, 32, rows, 128], &queries);
    let q = k.cast(&q, Dtype::Bf16).unwrap();
    let keys = import_f32(k, vec![1, 4, slots as u32, 128], &keys);
    let keys = k.cast(&keys, Dtype::Bf16).unwrap();
    let values = import_f32(k, vec![1, 4, slots as u32, 128], &values);
    let values = k.cast(&values, Dtype::Bf16).unwrap();
    let raw = [-0.03125; 32];
    let sinks = import_f32(k, vec![1, 32], &raw);
    let out = k
        .shared_attention(SharedAttention {
            q: &q,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: 2048 - rows,
            key_origin: origin,
            window: Some(513),
            sinks: Some(&sinks),
            scale: 1.0 / 128.0_f32.sqrt(),
        })
        .unwrap();
    assert_eq!(out.dtype(), Dtype::Bf16);
    assert_eq!(out.shape(), [1, 32, rows, 128]);
    let got = readback_f32(k, &out);
    assert_eq!(readback_f32(k, &sinks), raw);
    let mut last = Vec::with_capacity(32 * 128);
    for head in 0..32 {
        let start = (head * t + t - 1) * 128;
        last.extend_from_slice(&got[start..start + 128]);
    }
    last
}

#[test]
#[ignore = "bounded native same-query rounding diagnosis; no model weights"]
fn bf16_float_two_key_rounding() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    let a = two_key_union_last_row(&mut k, 512, 1024);
    let b = two_key_union_last_row(&mut k, 1, 1024);
    let c = two_key_union_last_row(&mut k, 1, 1535);
    let scale = f64::from(1.0 / 128.0_f32.sqrt());
    let reference = 1.0
        / (1.0 + (5.65625 * scale).exp() + (-0.03125_f64).exp() + 511.0 * (-1448.0 * scale).exp());
    let want = vec![reference; 32 * 128];
    let mut ratios = Vec::new();
    for got in [&a, &b, &c] {
        assert_eq!(got.len(), want.len());
        let (ratio, _) = pinned_max_ratio(got, &want, 32, 1, 128);
        assert!(ratio <= 2.0, "same-query exceeds pinned 2ULP: {ratio}");
        ratios.push(ratio);
    }
    let different =
        |left: &[f32], right: &[f32]| left.iter().zip(right).filter(|(a, b)| a != b).count();
    println!(
        "same-query A={} B={} C={} oracle={reference} ABdiff={} BCdiff={} ratios={ratios:?}",
        a[0],
        b[0],
        c[0],
        different(&a, &b),
        different(&b, &c)
    );
    let reference_host = import_f32(&mut k, vec![1], &[reference as f32]);
    let reference_bf16 = k.cast(&reference_host, Dtype::Bf16).unwrap();
    let rounded = readback_f32(&mut k, &reference_bf16)[0];
    assert_eq!(rounded, 71.0 / 256.0);
    // This closed-form cell must round once at output, not quantize P between tiles.
    for (name, got) in [("A", &a), ("B", &b), ("C", &c)] {
        assert!(
            got.iter().all(|&value| value == rounded),
            "{name}: expected {rounded}, first actual {}",
            got[0]
        );
    }
}
