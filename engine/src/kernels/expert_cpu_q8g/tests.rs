use super::*;
use crate::kernels::expert_cpu::{ExpertScratch, expert_forward};

/// Tiny deterministic LCG (Knuth MMIX constants); tests never touch the OS.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// Uniform in `[lo, hi)`; `next()` tops out at 2^53, so that is the scale.
    fn between(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
    }
}

/// `rows` Q4_K rows of `k` values: random bytes with a sane f16 `d` and `dmin` per block.
fn random_rows(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
    let mut v: Vec<u8> = (0..rows * k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES)
        .map(|_| rng.next() as u8)
        .collect();
    for blk in v.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
        let d = 0x2000 + (rng.next() % 0x1000) as u16;
        let dmin = 0x1c00 + (rng.next() % 0x1000) as u16;
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
    }
    v
}

fn quantized(x: &[f32]) -> Q8gRow {
    let mut row = Q8gRow::zeros(x.len()).unwrap();
    quantize_q8g(x, &mut row).unwrap();
    row
}

fn random_x(rng: &mut Lcg, k: usize) -> Vec<f32> {
    (0..k).map(|_| rng.between(-2.0, 3.0)).collect()
}

/// Decodes a Q4_K row to f64 values (independent of the kernels under test).
fn dequant_row(row: &[u8]) -> Vec<f64> {
    let f16 = |b: [u8; 2]| f64::from(f16_bits_to_f32(u16::from_le_bytes(b)));
    let mut out = Vec::new();
    for blk in row.as_chunks::<Q4K_BLOCK_BYTES>().0 {
        let (d, dmin) = (f16([blk[0], blk[1]]), f16([blk[2], blk[3]]));
        for j in 0..8 {
            let (sc, m) = scale_min(j, blk);
            for t in 0..GROUP {
                let byte = blk[16 + 32 * (j / 2) + t];
                let nib = if j % 2 == 1 { byte >> 4 } else { byte & 0x0f };
                out.push(d * f64::from(sc) * f64::from(nib) - dmin * f64::from(m));
            }
        }
    }
    out
}

#[test]
fn quantizer_round_trips_within_half_a_step_and_pins_the_edges() {
    let mut rng = Lcg(0x8E57_8E57_8E57_8E57);
    let k = 2 * Q4K_BLOCK_VALUES;
    let mut x = random_x(&mut rng, k);
    x[..GROUP].fill(0.0); // an all-zero group
    for (i, v) in x[GROUP..2 * GROUP].iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 }; // every |x| is the group's amax
    }
    x[2 * GROUP..3 * GROUP].fill(1e-30); // tiny but normal-scale, inv stays finite
    let a = quantized(&x);
    assert_eq!((a.d[0].to_bits(), a.s[0]), (0.0f32.to_bits(), 0));
    assert!(a.q[..GROUP].iter().all(|&q| q == 0));
    for g in 0..k / GROUP {
        if g == 0 {
            continue;
        }
        let xs = &x[g * GROUP..][..GROUP];
        let qs = &a.q[g * GROUP..][..GROUP];
        for (&v, &q) in xs.iter().zip(qs) {
            // d/2 plus a few ulp for the roundings in d, inv and x * inv.
            let bound = a.d[g] / 2.0 + a.d[g] * 128.0 * f32::EPSILON;
            assert!((v - a.d[g] * f32::from(q)).abs() <= bound, "group {g}");
        }
        assert_eq!(a.s[g], qs.iter().map(|&q| i32::from(q)).sum::<i32>());
        assert!(
            qs.iter().any(|&q| q.abs() == 127),
            "the amax element maps to +-127"
        );
    }
    let b = quantized(&x);
    assert_eq!(
        (
            a.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            &a.q,
            &a.s
        ),
        (
            b.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            &b.q,
            &b.s
        )
    );
}

#[test]
fn the_quantizer_refuses_bad_input_and_leaves_out_untouched() {
    let mut out = Q8gRow::zeros(Q4K_BLOCK_VALUES).unwrap();
    out.q.fill(7);
    out.s.fill(9);
    let good = vec![0.5f32; Q4K_BLOCK_VALUES];
    let mut nan = good.clone();
    nan[5] = f32::NAN;
    let mut inf = good.clone();
    inf[200] = f32::INFINITY;
    for bad in [nan.as_slice(), inf.as_slice()] {
        assert!(matches!(
            quantize_q8g(bad, &mut out),
            Err(ExpertCpuError::BadShape(_))
        ));
    }
    assert!(matches!(
        quantize_q8g(&good[..100], &mut out),
        Err(ExpertCpuError::BadShape(_))
    ));
    assert!(matches!(
        quantize_q8g(&[], &mut out),
        Err(ExpertCpuError::BadShape(_))
    ));
    let two = vec![0.5f32; 2 * Q4K_BLOCK_VALUES];
    assert!(matches!(
        quantize_q8g(&two, &mut out),
        Err(ExpertCpuError::BadLength { what: "out.d", .. })
    ));
    assert!(
        out.q.iter().all(|&q| q == 7) && out.s.iter().all(|&s| s == 9),
        "out untouched"
    );
    assert!(matches!(Q8gRow::zeros(0), Err(ExpertCpuError::BadShape(_))));
    assert!(matches!(
        Q8gRow::zeros(100),
        Err(ExpertCpuError::BadShape(_))
    ));
}

#[test]
fn scalar_group_dot_matches_an_independent_reference() {
    // Reference: decode the weights in f64 and dot them with the DEQUANTIZED activation
    // `d_g * q` in f64; the kernel only differs by its f32 rounding of the float tail.
    let mut rng = Lcg(0xD07_D07);
    for k in [Q4K_BLOCK_VALUES, 2 * Q4K_BLOCK_VALUES, 8 * Q4K_BLOCK_VALUES] {
        let w = random_rows(&mut rng, 16, k);
        let xq = quantized(&random_x(&mut rng, k));
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        for row in w.chunks_exact(row_bytes) {
            let deq = dequant_row(row);
            let (mut want, mut mag) = (0.0f64, 0.0f64);
            for (i, wv) in deq.iter().enumerate() {
                let term = wv * f64::from(xq.d[i / GROUP]) * f64::from(xq.q[i]);
                want += term;
                mag += term.abs();
            }
            let got = f64::from(q4k_q8g_row_dot_scalar(row, &xq));
            assert!(
                (got - want).abs() <= 1e-5 * mag + 1e-9,
                "k {k}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn q8g_error_stays_inside_the_derived_bound_against_the_f32_reference() {
    // Each activation moves by at most d_g / 2, so a row moves by at most
    // sum_g (d_g / 2) * sum_{i in g} |w_i|; plus the f32 tail, 1e-5 of the magnitude.
    let mut rng = Lcg(0xB0B0_B0B0);
    for k in [2048usize, 512] {
        let w = random_rows(&mut rng, 64, k);
        let x = random_x(&mut rng, k);
        let xq = quantized(&x);
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        for row in w.chunks_exact(row_bytes) {
            let deq = dequant_row(row);
            let exact: f64 = deq.iter().zip(&x).map(|(w, x)| w * f64::from(*x)).sum();
            let mag: f64 = deq
                .iter()
                .zip(&x)
                .map(|(w, x)| (w * f64::from(*x)).abs())
                .sum();
            let abs_w: Vec<f64> = deq.iter().map(|w| w.abs()).collect();
            let bound: f64 = abs_w
                .as_chunks::<GROUP>()
                .0
                .iter()
                .zip(&xq.d)
                .map(|(ws, d)| f64::from(*d) / 2.0 * ws.iter().sum::<f64>())
                .sum();
            let got = f64::from(q4k_q8g_row_dot_scalar(row, &xq));
            assert!((got - exact).abs() <= bound + 1e-5 * mag + 1e-9, "k {k}");
        }
    }
}

/// Rows built to hit the corners of the integer ranges: every nibble 15 with the largest scales
/// and mins, every q = +-127, alternating signs.
fn extreme_rows(k: usize, rows: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
    let mut w = vec![0xFFu8; rows * k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES];
    for blk in w.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let alternating: Vec<f32> = (0..k)
        .map(|i| if i % 2 == 0 { 127.0 } else { -127.0 })
        .collect();
    let positive = vec![5.0f32; k];
    let negative = vec![-3.0f32; k];
    (w, vec![alternating, positive, negative])
}

/// On a host without AVX2 this prints "skipped" and passes, because then the scalar path is the
/// only path there is; on a host with it, it compares every bit.
#[test]
fn avx2_equals_scalar_bitwise() {
    if DotBackend::detect() == DotBackend::scalar() {
        println!("avx2_equals_scalar_bitwise: AVX2 not detected, skipped");
        return;
    }
    let mut rng = Lcg(0xA5A5_5A5A);
    let mut cases: Vec<(usize, Vec<u8>, Vec<Vec<f32>>)> = Vec::new();
    for k in [Q4K_BLOCK_VALUES, 2 * Q4K_BLOCK_VALUES, 2048] {
        for rows in [3usize, 64] {
            let xs = (0..4).map(|_| random_x(&mut rng, k)).collect();
            cases.push((k, random_rows(&mut rng, rows, k), xs));
        }
        let (w, xs) = extreme_rows(k, 3);
        cases.push((k, w, xs));
    }
    for (k, w, xs) in cases {
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let rows = w.len() / row_bytes;
        for x in xs {
            let xq = quantized(&x);
            let (mut want, mut got) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            q4k_q8g_matvec(DotBackend::scalar(), &w, rows, k, &xq, &mut want).unwrap();
            q4k_q8g_matvec(DotBackend::detect(), &w, rows, k, &xq, &mut got).unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "k {k} rows {rows}");
        }
    }
}

#[test]
fn matvec_validates_before_touching_out() {
    let mut rng = Lcg(3);
    let k = Q4K_BLOCK_VALUES;
    let w = random_rows(&mut rng, 2, k);
    let xq = quantized(&random_x(&mut rng, k));
    let backend = DotBackend::scalar();
    let mut out = vec![7.0f32; 2];
    let bad_shape = |r: Result<(), ExpertCpuError>| matches!(r, Err(ExpertCpuError::BadShape(_)));
    assert!(bad_shape(q4k_q8g_matvec(
        backend, &w, 2, 100, &xq, &mut out
    )));
    assert!(bad_shape(q4k_q8g_matvec(backend, &w, 0, k, &xq, &mut out)));
    assert!(matches!(
        q4k_q8g_matvec(backend, &w[1..], 2, k, &xq, &mut out),
        Err(ExpertCpuError::BadLength { what: "w", .. })
    ));
    assert!(matches!(
        q4k_q8g_matvec(backend, &w, 2, k, &xq, &mut out[..1]),
        Err(ExpertCpuError::BadLength { what: "out", .. })
    ));
    let short = Q8gRow::zeros(2 * k).unwrap();
    assert!(matches!(
        q4k_q8g_matvec(backend, &w, 2, k, &short, &mut out),
        Err(ExpertCpuError::BadLength { what: "xq.d", .. })
    ));
    assert_eq!(out, [7.0, 7.0], "out untouched by every refusal");
    q4k_q8g_matvec(backend, &w, 2, k, &xq, &mut out).unwrap();
}

const SHAPE: ExpertShape = ExpertShape {
    hidden: 512,
    inter: 256,
};

fn expert_bytes(rng: &mut Lcg) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        random_rows(rng, SHAPE.inter, SHAPE.hidden),
        random_rows(rng, SHAPE.inter, SHAPE.hidden),
        random_rows(rng, SHAPE.hidden, SHAPE.inter),
    )
}

#[test]
fn expert_forward_q8g_is_close_to_the_f32_expert() {
    // Measured on this fixture (random bytes, normal-like x): the relative L2 error of the
    // output against the f32 expert is the activation grid's own noise, about 1 percent; the
    // bound leaves 5x headroom for a different seed.
    let mut rng = Lcg(0xFEED);
    let (g, u, d) = expert_bytes(&mut rng);
    let slices = ExpertSlices {
        gate: &g,
        up: &u,
        down: &d,
    };
    let x = random_x(&mut rng, SHAPE.hidden);
    let (mut want, mut got) = (vec![0.0f32; SHAPE.hidden], vec![0.0f32; SHAPE.hidden]);
    expert_forward(
        SHAPE,
        &slices,
        &x,
        &mut ExpertScratch::new(SHAPE),
        &mut want,
    )
    .unwrap();
    let mut scratch = ExpertScratchQ8g::new(SHAPE);
    expert_forward_q8g(
        DotBackend::detect(),
        SHAPE,
        &slices,
        &x,
        &mut scratch,
        &mut got,
    )
    .unwrap();
    let num: f64 = want
        .iter()
        .zip(&got)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let den: f64 = want.iter().map(|a| f64::from(*a).powi(2)).sum();
    let rel = (num / den).sqrt();
    assert!(rel < 0.05, "relative L2 error {rel}");
}

#[test]
fn expert_forward_q8g_validates_like_the_f32_path() {
    let mut rng = Lcg(0xBAD);
    let (g, u, d) = expert_bytes(&mut rng);
    let slices = ExpertSlices {
        gate: &g,
        up: &u,
        down: &d,
    };
    let x = random_x(&mut rng, SHAPE.hidden);
    let backend = DotBackend::scalar();
    let mut scratch = ExpertScratchQ8g::new(SHAPE);
    let mut out = vec![7.0f32; SHAPE.hidden];
    let odd = ExpertShape {
        hidden: 100,
        inter: 256,
    };
    assert!(matches!(
        expert_forward_q8g(backend, odd, &slices, &x, &mut scratch, &mut out),
        Err(ExpertCpuError::BadShape(_))
    ));
    for (what, run) in [
        (
            "x",
            expert_forward_q8g(backend, SHAPE, &slices, &x[1..], &mut scratch, &mut out),
        ),
        (
            "gate",
            expert_forward_q8g(
                backend,
                SHAPE,
                &ExpertSlices {
                    gate: &g[1..],
                    ..slices
                },
                &x,
                &mut scratch,
                &mut out,
            ),
        ),
        (
            "down",
            expert_forward_q8g(
                backend,
                SHAPE,
                &ExpertSlices {
                    down: &d[1..],
                    ..slices
                },
                &x,
                &mut scratch,
                &mut out,
            ),
        ),
    ] {
        assert!(
            matches!(run, Err(ExpertCpuError::BadLength { what: w, .. }) if w == what),
            "{what}"
        );
    }
    let mut wrong = ExpertScratchQ8g::new(ExpertShape {
        hidden: 512,
        inter: 512,
    });
    assert!(matches!(
        expert_forward_q8g(backend, SHAPE, &slices, &x, &mut wrong, &mut out),
        Err(ExpertCpuError::BadLength {
            what: "scratch.gate",
            ..
        })
    ));
    let mut nan = x.clone();
    nan[3] = f32::NAN;
    assert!(expert_forward_q8g(backend, SHAPE, &slices, &nan, &mut scratch, &mut out).is_err());
    assert!(
        out.iter().all(|&v| v == 7.0),
        "out untouched on every error"
    );
    // The scratch is reusable after the refusals and the result is deterministic.
    let (mut a, mut b) = (vec![0.0f32; SHAPE.hidden], vec![0.0f32; SHAPE.hidden]);
    expert_forward_q8g(backend, SHAPE, &slices, &x, &mut scratch, &mut a).unwrap();
    expert_forward_q8g(backend, SHAPE, &slices, &x, &mut scratch, &mut b).unwrap();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// Skips (prints, passes) on a host without AVX2, like `avx2_equals_scalar_bitwise`.
#[test]
fn scalar_and_avx2_expert_forward_are_bitwise_equal() {
    if DotBackend::detect() == DotBackend::scalar() {
        println!("scalar_and_avx2_expert_forward_are_bitwise_equal: AVX2 not detected, skipped");
        return;
    }
    let mut rng = Lcg(0x7E57);
    for _ in 0..4 {
        let (g, u, d) = expert_bytes(&mut rng);
        let slices = ExpertSlices {
            gate: &g,
            up: &u,
            down: &d,
        };
        let x = random_x(&mut rng, SHAPE.hidden);
        let (mut a, mut b) = (vec![0.0f32; SHAPE.hidden], vec![0.0f32; SHAPE.hidden]);
        let mut s1 = ExpertScratchQ8g::new(SHAPE);
        let mut s2 = ExpertScratchQ8g::new(SHAPE);
        expert_forward_q8g(DotBackend::scalar(), SHAPE, &slices, &x, &mut s1, &mut a).unwrap();
        expert_forward_q8g(DotBackend::detect(), SHAPE, &slices, &x, &mut s2, &mut b).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn fast_round_matches_reference_at_every_half_boundary_and_extreme_scale() {
    use super::{Q8gRow, fill_groups, round_quantized};
    for n in -128..=128 {
        let half = n as f32 + 0.5;
        for x in [half.next_down(), half, half.next_up(), n as f32] {
            assert_eq!(round_quantized(x), x.round().clamp(-127.0, 127.0) as i8);
        }
    }
    for scale in [0.0, f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX] {
        let input: Vec<_> = (0..256)
            .map(|i| ((i % 33) as f32 / 16.0 - 1.0) * scale)
            .collect();
        let mut reference = Q8gRow::zeros(input.len()).unwrap();
        let mut fast = Q8gRow::zeros(input.len()).unwrap();
        fill_groups::<false>(&input, &mut reference);
        fill_groups::<true>(&input, &mut fast);
        assert_eq!(reference.q, fast.q);
        assert_eq!(reference.s, fast.s);
        assert!(
            reference
                .d
                .iter()
                .zip(fast.d)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

#[test]
fn narrow_sum_preserves_each_group_at_signed_byte_extremes() {
    if DotBackend::detect() == DotBackend::scalar() {
        return;
    }
    // The selector is cached once per process. Exercise the opt-in under
    // ordinary CI as well as when the entire test binary runs with it set.
    if std::env::var_os("FIFI_CPU_Q8G_NARROW_SUM").is_none_or(|v| v != "1") {
        let case = "kernels::expert_cpu_q8g::tests::narrow_sum_preserves_each_group_at_signed_byte_extremes";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case])
            .env("FIFI_CPU_Q8G_NARROW_SUM", "1")
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success() && log.contains("1 passed"), "{log}");
        return;
    }

    let (mut row, _) = extreme_rows(Q4K_BLOCK_VALUES, 1);
    row[2..4].fill(0); // Isolate the product sum from the minimum correction.
    let mut xq = Q8gRow::zeros(Q4K_BLOCK_VALUES).unwrap();
    for active in 0..GROUPS_PER_BLOCK {
        xq.d.fill(0.0);
        xq.d[active] = 1.0;
        for extreme in [i8::MIN, i8::MAX] {
            xq.q.fill(0);
            xq.q[active * GROUP..(active + 1) * GROUP].fill(extreme);
            let mut got = [0.0];
            q4k_q8g_matvec(
                DotBackend::detect(),
                &row,
                1,
                Q4K_BLOCK_VALUES,
                &xq,
                &mut got,
            )
            .unwrap();
            let expected = f32::from(extreme) * (32 * 15 * 63) as f32;
            assert_eq!(
                got[0].to_bits(),
                expected.to_bits(),
                "group {active}, q {extreme}"
            );
        }
    }
}

/// Bit-for-bit fast-vs-reference over the dense set. Every scale is a real
/// float with a full mantissa (bit patterns drawn from the whole finite
/// range), every half boundary k+0.5 is swept through each scale, and a
/// million-value sweep draws full random mantissas across the whole finite
/// range - the earlier version of this test multiplied a smallest-subnormal
/// seed by sqrt(2), which collapsed to 2^-149 on every step, so its scales
/// had no mantissa spread at all and compared the reference with itself.
#[test]
fn fast_quantization_is_bit_identical_dense() {
    use super::{Q8gRow, fill_groups};
    let assert_pair = |values: &[f32]| {
        // The row contract wants a 256-value multiple; pad with zeros.
        let mut input = values.to_vec();
        input.resize(values.len().div_ceil(256) * 256, 0.0);
        let input = &input[..];
        let mut reference = Q8gRow::zeros(input.len()).unwrap();
        let mut fast = Q8gRow::zeros(input.len()).unwrap();
        fill_groups::<false>(input, &mut reference);
        fill_groups::<true>(input, &mut fast);
        assert_eq!(reference.q, fast.q, "q bytes differ at {input:?}");
        assert_eq!(reference.s, fast.s);
        assert!(
            reference
                .d
                .iter()
                .zip(&fast.d)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "scales differ"
        );
    };
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    // Scales: 145 random FINITE bit patterns (any exponent from subnormal to
    // max, full random mantissa) plus the exact edges. Multiplying a half
    // boundary by such a scale produces values whose quantized position
    // depends on every mantissa bit, not just the exponent.
    let mut scales = vec![f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX];
    for _ in 0..145 {
        let bits = (lcg.next() as u32) & 0x7F7F_FFFF; // finite, never NaN/inf
        let bits = bits | 1; // never exactly zero
        scales.push(f32::from_bits(bits));
    }
    for &s in &scales {
        for n in -128i32..=128 {
            let half = n as f32 + 0.5;
            let input: Vec<f32> = [half.next_down(), half, half.next_up()]
                .iter()
                .map(|&v| v * s)
                .collect();
            assert_pair(&input);
            // With the group maximum at 127 * s the reciprocal scale maps s to 1, so
            // the same values land on (or one ulp either side of) the k + 0.5
            // quantization boundaries; alone in the group they quantize to +-127.
            let top = 127.0 * s;
            if top.is_finite() {
                let mut with_max = vec![top];
                with_max.extend_from_slice(&input);
                assert_pair(&with_max);
            }
        }
        // Saturation edges: the group max itself and values beyond it in magnitude.
        assert_pair(&[
            0.0,
            -0.0,
            s,
            -s,
            127.0 * s,
            -127.0 * s,
            128.0 * s,
            -128.0 * s,
        ]);
    }
    // Corners the boundary sweep cannot hit: negative zero, the smallest
    // subnormal, and a reciprocal-overflowing group (amax subnormal makes
    // 127/amax infinite).
    assert_pair(&[-0.0; 256]);
    assert_pair(&[f32::from_bits(1); 256]);
    assert_pair(&[
        f32::from_bits(1),
        0.0,
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.0,
        -1.0,
        f32::from_bits(1) * 2.0,
        0.0,
    ]);
    // Seeded sweep: one million values with FULL random mantissas over the
    // whole finite range, both signs, subnormals included - drawn from raw
    // bit patterns so no mantissa bit is pinned.
    let mut input = Vec::with_capacity(1_000_000);
    for _ in 0..1_000_000 / 2 {
        let magnitude = f32::from_bits((lcg.next() as u32) & 0x7F7F_FFFF);
        let signed = if lcg.next() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        input.push(signed);
    }
    for chunk in input.chunks(256) {
        assert_pair(chunk);
    }
}

/// The default with the variable unset, pinned in a child process: the
/// `OnceLock` is process-global, so a parent that has already quantized
/// once cannot observe the unset arm. The child (fresh process, the
/// variable removed) quantizes the fixed row through the public entry and
/// prints its digest; the parent asserts it equals the forced fast digest
/// on an AVX2 host and the forced reference digest everywhere else, and
/// that every rollback spelling disables while the enable words keep the
/// default.
#[test]
fn fast_quantizer_default_matches_forced_path_in_a_child_process() {
    use super::{Q8gRow, fill_groups, quantize_q8g};
    const CHILD: &str = "kernels::expert_cpu_q8g::tests::fast_quantizer_default_matches_forced_path_in_a_child_process";

    fn digest(row: &Q8gRow) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in row.q.iter() {
            h ^= *byte as u8 as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        for word in row.s.iter() {
            h ^= u64::from(*word as u32);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        for f in &row.d {
            h ^= f.to_bits() as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }

    let mut lcg = Lcg(0x0123_4567_89ab_cdef);
    let input: Vec<f32> = (0..768)
        .map(|_| f32::from_bits(((lcg.next() as u32) & 0x7f7f_ffff) | 1))
        .collect();

    let forced = |fast: bool| {
        let mut out = Q8gRow::zeros(input.len()).unwrap();
        if fast {
            fill_groups::<true>(&input, &mut out);
        } else {
            fill_groups::<false>(&input, &mut out);
        }
        digest(&out)
    };
    let reference = forced(false);
    let fast = forced(true);

    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    let expected_default = if avx2 { fast } else { reference };

    // The digest channel is a file: libtest swallows a passing test's stdout
    // under capture, so a print cannot be read back by the parent.
    let out_path = std::env::temp_dir().join("fifi-fast-quant-child-digest");
    let run_child = |value: Option<&str>| -> u64 {
        let _ = std::fs::remove_file(&out_path);
        let mut command = std::process::Command::new(std::env::current_exe().expect("test binary"));
        command
            .args(["--exact", CHILD, "--test-threads", "1"])
            .env("FQ_CHILD", "1")
            .env("FQ_CHILD_OUT", &out_path)
            .env_remove("FIFI_CPU_Q8G_FAST_QUANT");
        if let Some(value) = value {
            command.env("FIFI_CPU_Q8G_FAST_QUANT", value);
        }
        let output = command.output().expect("spawn the fast-quant child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "fast-quant child {value:?} exited with {}\nstdout:\n{stdout}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read_to_string(&out_path)
            .unwrap_or_else(|_| panic!("child {value:?} wrote no digest file:\n{stdout}"))
            .trim()
            .parse()
            .expect("digest digits")
    };

    if let Some(path) = std::env::var_os("FQ_CHILD_OUT") {
        let mut out = Q8gRow::zeros(input.len()).unwrap();
        quantize_q8g(&input, &mut out).expect("row contract");
        std::fs::write(path, digest(&out).to_string()).expect("write the digest file");
        return;
    }
    assert_eq!(
        run_child(None),
        expected_default,
        "unset default must take the {} path",
        if avx2 { "fast" } else { "reference" }
    );
    for rollback in ["0", "off", "false", "maybe", ""] {
        assert_eq!(
            run_child(Some(rollback)),
            reference,
            "rollback {rollback:?}"
        );
    }
    for enable in ["1", "true", "on"] {
        assert_eq!(
            run_child(Some(enable)),
            expected_default,
            "enable {enable:?} must keep the default path"
        );
    }
}

/// Seeded random sweep: the narrow default against the forced-wide path over full-range
/// quantized activations and full-range weight bytes, compared across two processes (the
/// selector caches once per process): this process runs the default (narrow), a child
/// with FIFI_CPU_Q8G_NARROW_SUM=0 runs the wide path, and both print the same digest of
/// every result. Both are exact integer sums of the same products, so the digests match.
#[test]
fn narrow_default_matches_forced_wide_over_random_sweep() {
    if DotBackend::detect() == DotBackend::scalar() {
        return;
    }
    let digest_of = || {
        let mut lcg = Lcg(0xD1B5_4A32_D192_ED03);
        let mut hasher: [u64; 4] = [0; 4];
        let mut row = vec![0u8; Q4K_BLOCK_BYTES];
        let mut xq = Q8gRow::zeros(Q4K_BLOCK_VALUES).unwrap();
        let mut out = [0.0f32; 1];
        let groups = Q4K_BLOCK_VALUES / GROUP;
        for _ in 0..2_000 {
            for b in &mut row {
                *b = (lcg.next() >> 24) as u8;
            }
            for g in 0..groups {
                xq.d[g] = ((lcg.next() % 2001) as f32 - 1000.0) / 1_000_000.0;
                for q in &mut xq.q[g * GROUP..(g + 1) * GROUP] {
                    *q = (lcg.next() >> 45) as i8;
                }
                xq.s[g] = xq.q[g * GROUP..(g + 1) * GROUP]
                    .iter()
                    .map(|&q| i32::from(q))
                    .sum();
            }
            q4k_q8g_matvec(
                DotBackend::detect(),
                &row,
                1,
                Q4K_BLOCK_VALUES,
                &xq,
                &mut out,
            )
            .unwrap();
            for (h, bytes) in hasher.iter_mut().zip(out[0].to_le_bytes().chunks(2)) {
                *h = h
                    .rotate_left(17)
                    .wrapping_add(u64::from(bytes[0]) | (u64::from(bytes[1]) << 8));
            }
        }
        hasher
    };
    let here = digest_of();
    let own = format!("{here:02x?}");
    if let Some(reply) = std::env::var_os("Q8G_SWEEP_REPLY") {
        // Child with FIFI_CPU_Q8G_NARROW_SUM=0: hand the digest back through the file
        // the parent named, immune to the harness output capture.
        std::fs::write(reply, &own).unwrap();
        return;
    }
    if std::env::var_os("FIFI_CPU_Q8G_NARROW_SUM").is_none_or(|v| v != "0") {
        let case =
            "kernels::expert_cpu_q8g::tests::narrow_default_matches_forced_wide_over_random_sweep";
        let reply = std::fs::canonicalize(std::env::temp_dir())
            .unwrap_or_else(|_| std::env::current_dir().unwrap())
            .join(format!("q8g-sweep-{}.txt", std::process::id()));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case])
            .env("FIFI_CPU_Q8G_NARROW_SUM", "0")
            .env("Q8G_SWEEP_REPLY", &reply)
            .output()
            .unwrap();
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{log}");
        let forced = std::fs::read_to_string(&reply).unwrap_or_default();
        let _ = std::fs::remove_file(&reply);
        assert_eq!(
            own,
            forced.trim(),
            "narrow default and forced-wide digests differ; child log: {log}"
        );
    }
}

/// Both dot paths FORCED and compared bit for bit: a vacuous comparison
/// (wide with wide, which happens if the kernel default never flips) cannot
/// pass here. Random full-range rows and quantized activations through
/// `q4k_q8g_row_dot_forced::<true>` and `::<false>`, equal `to_bits`. Runs on
/// every x86_64 host (the helper falls back to the scalar dot without AVX2,
/// where both sides are the same scalar call by construction); the AVX2
/// module compiles to an empty stub on other targets, where both paths are
/// the scalar reference.
#[cfg(target_arch = "x86_64")]
#[test]
fn narrow_and_wide_forced_paths_are_bit_identical_over_random_rows() {
    use crate::kernels::expert_cpu_avx2::q4k_q8g_row_dot_forced;
    let mut lcg = Lcg(0xA5A5_5A5A_1234_9876);
    let mut row = vec![0u8; Q4K_BLOCK_BYTES];
    let mut xq = Q8gRow::zeros(Q4K_BLOCK_VALUES).unwrap();
    let groups = Q4K_BLOCK_VALUES / GROUP;
    for _ in 0..4_000 {
        for b in &mut row {
            *b = (lcg.next() >> 24) as u8;
        }
        for g in 0..groups {
            xq.d[g] = ((lcg.next() % 2001) as f32 - 1000.0) / 1_000_000.0;
            for q in &mut xq.q[g * GROUP..(g + 1) * GROUP] {
                *q = (lcg.next() >> 45) as i8;
            }
            xq.s[g] = xq.q[g * GROUP..(g + 1) * GROUP]
                .iter()
                .map(|&q| i32::from(q))
                .sum();
        }
        let narrow = q4k_q8g_row_dot_forced::<true>(&row, &xq);
        let wide = q4k_q8g_row_dot_forced::<false>(&row, &xq);
        assert_eq!(
            narrow.to_bits(),
            wide.to_bits(),
            "forced paths differ on row {row:?}"
        );
    }
}

/// Activation rows for the identity tests: real-sized random rows, and rows built to hit the
/// corners of the quantizer (constants at the signed byte ends, zeros, negative zeros, one
/// large outlier in every group, values on exact half boundaries under a unit scale).
fn identity_activation_rows(k: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg(0x1DE7_0000 ^ k as u64);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for _ in 0..6 {
        // Sum of four uniforms: a bell shape like real hidden activations.
        rows.push(
            (0..k)
                .map(|_| (0..4).map(|_| rng.between(-1.0, 1.0)).sum::<f32>())
                .collect(),
        );
    }
    rows.push(vec![127.0; k]);
    rows.push(vec![-128.0; k]);
    rows.push(vec![0.0; k]);
    rows.push(vec![-0.0; k]);
    rows.push(
        (0..k)
            .map(|i| if i % 2 == 0 { 128.0 } else { -128.0 })
            .collect(),
    );
    for outlier in [1.0e4f32, -1.0e4, 3.0e38, -3.0e38, 1.0e-30] {
        rows.push(
            (0..k)
                .map(|i| {
                    if i % GROUP == (i / GROUP * 7) % GROUP {
                        outlier
                    } else {
                        rng.between(-1.0, 1.0) * outlier.abs().min(1.0)
                    }
                })
                .collect(),
        );
    }
    // Unit scale (the group maximum is exactly 127) with every other value on a half boundary.
    rows.push(
        (0..k)
            .map(|i| {
                if i % GROUP == 0 {
                    127.0
                } else {
                    (rng.next() % 126) as f32 - 62.5
                }
            })
            .collect(),
    );
    rows
}

/// Already-quantized activation rows at the byte extremes, with the group sums the quantizer
/// would store. The quantizer never emits -128, but the kernel accepts it, so the bound has to
/// hold there too.
#[cfg(target_arch = "x86_64")]
fn identity_quantized_rows(k: usize) -> Vec<Q8gRow> {
    let mut rng = Lcg(0xFEED_0000 ^ k as u64);
    let mut rows = Vec::new();
    let mut push = |f: &mut dyn FnMut(usize) -> i8| {
        let mut row = Q8gRow::zeros(k).unwrap();
        for g in 0..k / GROUP {
            row.d[g] = 0.0078125 * (1 + g % 5) as f32;
            for t in 0..GROUP {
                row.q[g * GROUP + t] = f(g * GROUP + t);
            }
            row.s[g] = row.q[g * GROUP..(g + 1) * GROUP]
                .iter()
                .map(|&q| i32::from(q))
                .sum();
        }
        rows.push(row);
    };
    push(&mut |_| -128);
    push(&mut |_| 127);
    push(&mut |_| 0);
    push(&mut |i| if i % 2 == 0 { 127 } else { -128 });
    push(&mut |i| {
        if i % GROUP == (i / GROUP * 5) % GROUP {
            127
        } else {
            0
        }
    });
    push(&mut |i| {
        if i % GROUP == (i / GROUP * 5) % GROUP {
            -128
        } else {
            0
        }
    });
    push(&mut |_| (rng.next() >> 45) as i8);
    rows
}

/// Weight rows for the identity tests: random bytes with sane block scales, nibbles all 15
/// with the largest sub-block scales and mins, and nibbles all 0.
fn identity_weight_rows(k: usize, rows: usize) -> Vec<Vec<u8>> {
    let mut rng = Lcg(0xB10C_0000 ^ k as u64);
    let blocks = rows * k / Q4K_BLOCK_VALUES;
    let mut zero = vec![0u8; blocks * Q4K_BLOCK_BYTES];
    for blk in zero.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        blk[4..16].fill(0xFF);
    }
    let mut high = vec![0xFFu8; blocks * Q4K_BLOCK_BYTES];
    for blk in high.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    vec![
        random_rows(&mut rng, rows, k),
        random_rows(&mut rng, rows, k),
        high,
        zero,
    ]
}

fn fnv_bits(h: &mut u64, v: u64) {
    *h ^= v;
    *h = h.wrapping_mul(0x1000_0000_01b3);
}

/// The quantizer's two paths on every identity row, bit for bit (`q`, `s`, `d`).
#[test]
fn identity_quantizer_fast_equals_reference_on_real_sized_and_extreme_rows() {
    for k in [512usize, 2048, 4096] {
        for (n, x) in identity_activation_rows(k).iter().enumerate() {
            let mut reference = Q8gRow::zeros(k).unwrap();
            let mut fast = Q8gRow::zeros(k).unwrap();
            fill_groups::<false>(x, &mut reference);
            fill_groups::<true>(x, &mut fast);
            assert_eq!(reference.q, fast.q, "q differs: k {k} row {n}");
            assert_eq!(reference.s, fast.s, "s differs: k {k} row {n}");
            let bits = |d: &[f32]| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&reference.d),
                bits(&fast.d),
                "d differs: k {k} row {n}"
            );
        }
    }
}

/// The expert dot, three ways, on every identity weight row and activation row: the scalar
/// reference, the AVX2 dot with wide sums, and the AVX2 dot with 16-bit sums, all bit-equal.
#[cfg(target_arch = "x86_64")]
#[test]
fn identity_dot_scalar_wide_and_narrow_agree_on_real_sized_and_extreme_rows() {
    use crate::kernels::expert_cpu_avx2::q4k_q8g_row_dot_forced;
    if DotBackend::detect() == DotBackend::scalar() {
        println!("AVX2 not detected, skipped");
        return;
    }
    let mut compared = 0usize;
    for k in [512usize, 2048] {
        let mut activations: Vec<Q8gRow> = identity_quantized_rows(k);
        for x in identity_activation_rows(k) {
            activations.push(quantized(&x));
        }
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        for weights in identity_weight_rows(k, 128) {
            for xq in &activations {
                for row in weights.chunks_exact(row_bytes) {
                    let want = q4k_q8g_row_dot_scalar(row, xq);
                    let wide = q4k_q8g_row_dot_forced::<false>(row, xq);
                    let narrow = q4k_q8g_row_dot_forced::<true>(row, xq);
                    assert_eq!(wide.to_bits(), want.to_bits(), "wide vs scalar, k {k}");
                    assert_eq!(narrow.to_bits(), want.to_bits(), "narrow vs scalar, k {k}");
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 10_000, "only {compared} rows compared");
}

/// Digest of the full public path on the identity inputs: the quantizer entry (`quantize_q8g`,
/// which reads the fast-path switch), the matvec on the detected backend (which reads the
/// narrow-sum switch), and a real-sized expert forward.
fn identity_pipeline_digest() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let backend = DotBackend::detect();
    let shape = ExpertShape {
        hidden: 2048,
        inter: 512,
    };
    let rows = 64usize;
    for k in [512usize, 2048] {
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let weights = identity_weight_rows(k, rows);
        for x in identity_activation_rows(k) {
            let mut xq = Q8gRow::zeros(k).unwrap();
            quantize_q8g(&x, &mut xq).unwrap();
            for b in &xq.q {
                fnv_bits(&mut h, *b as u8 as u64);
            }
            for v in &xq.d {
                fnv_bits(&mut h, u64::from(v.to_bits()));
            }
            for s in &xq.s {
                fnv_bits(&mut h, u64::from(*s as u32));
            }
            for w in &weights {
                debug_assert_eq!(w.len(), rows * row_bytes);
                let mut out = vec![0.0f32; rows];
                q4k_q8g_matvec(backend, w, rows, k, &xq, &mut out).unwrap();
                for v in out {
                    fnv_bits(&mut h, u64::from(v.to_bits()));
                }
            }
        }
    }
    {
        let mut rng = Lcg(0xE4E4_0001);
        let gate = random_rows(&mut rng, shape.inter, shape.hidden);
        let up = random_rows(&mut rng, shape.inter, shape.hidden);
        let down = random_rows(&mut rng, shape.hidden, shape.inter);
        let slices = ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down,
        };
        let mut scratch = ExpertScratchQ8g::new(shape);
        for x in identity_activation_rows(shape.hidden)
            .into_iter()
            .filter(|x| x.iter().all(|v| v.abs() < 1.0e6))
        {
            let mut out = vec![0.0f32; shape.hidden];
            expert_forward_q8g(backend, shape, &slices, &x, &mut scratch, &mut out).unwrap();
            for v in out {
                fnv_bits(&mut h, u64::from(v.to_bits()));
            }
        }
    }
    h
}

/// The two kernel switches through the public path, in child processes (each switch is read
/// once per process): every combination of unset, rollback words and enable words yields the
/// same digest as the child with both switches rolled back.
#[test]
fn identity_public_path_is_bit_equal_under_every_switch_setting() {
    const CHILD: &str = "kernels::expert_cpu_q8g::tests::identity_public_path_is_bit_equal_under_every_switch_setting";
    if let Some(path) = std::env::var_os("IDENTITY_CHILD_OUT") {
        std::fs::write(path, identity_pipeline_digest().to_string()).unwrap();
        return;
    }
    let out_path = std::env::temp_dir().join(format!("fifi-identity-{}", std::process::id()));
    let run = |fast: Option<&str>, narrow: Option<&str>| -> u64 {
        let _ = std::fs::remove_file(&out_path);
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", CHILD, "--test-threads", "1"])
            .env("IDENTITY_CHILD_OUT", &out_path)
            .env_remove("FIFI_CPU_Q8G_FAST_QUANT")
            .env_remove("FIFI_CPU_Q8G_NARROW_SUM");
        if let Some(v) = fast {
            command.env("FIFI_CPU_Q8G_FAST_QUANT", v);
        }
        if let Some(v) = narrow {
            command.env("FIFI_CPU_Q8G_NARROW_SUM", v);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "child {fast:?}/{narrow:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read_to_string(&out_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    };
    let rolled_back = run(Some("0"), Some("0"));
    for (fast, narrow) in [
        (None, None),
        (Some("1"), Some("1")),
        (Some("0"), None),
        (None, Some("0")),
        (Some("off"), Some("false")),
        (Some("garbage"), Some("")),
    ] {
        assert_eq!(
            run(fast, narrow),
            rolled_back,
            "fast {fast:?} narrow {narrow:?} differs from both switches off"
        );
    }
    let _ = std::fs::remove_file(&out_path);
}

/// The public quantizer and the detected dot against the path the switches fall back to
/// (reference quantizer, scalar dot), on the identity rows, in this process.
#[test]
fn identity_default_quantizer_and_matvec_equal_the_scalar_reference() {
    let k = 2048usize;
    let rows = 64usize;
    let weights = identity_weight_rows(k, rows);
    for x in identity_activation_rows(k) {
        let mut want = Q8gRow::zeros(k).unwrap();
        fill_groups::<false>(&x, &mut want);
        let got = quantized(&x);
        assert_eq!(got.q, want.q);
        assert_eq!(got.s, want.s);
        for w in &weights {
            let (mut a, mut b) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            q4k_q8g_matvec(DotBackend::scalar(), w, rows, k, &want, &mut a).unwrap();
            q4k_q8g_matvec(DotBackend::detect(), w, rows, k, &got, &mut b).unwrap();
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

/// The split shape is BIT-IDENTICAL to the whole forward: halves + the down pass
/// on the joining thread reproduce `expert_forward_q8g`'s output exactly (gate and
/// up rows are independent integer dots; the down pass is the same arithmetic on
/// the same assembled act). The split-job lever's correctness floor.
#[test]
fn split_halves_plus_down_match_the_whole_forward_bitwise() {
    let mut rng = Lcg(0x5B17);
    for _ in 0..4 {
        let (g, u, d) = expert_bytes(&mut rng);
        let slices = ExpertSlices {
            gate: &g,
            up: &u,
            down: &d,
        };
        let x = random_x(&mut rng, SHAPE.hidden);
        let mut whole = vec![0.0f32; SHAPE.hidden];
        let mut scratch = ExpertScratchQ8g::new(SHAPE);
        expert_forward_q8g(
            DotBackend::scalar(),
            SHAPE,
            &slices,
            &x,
            &mut scratch,
            &mut whole,
        )
        .unwrap();

        let mid = SHAPE.inter / 2;
        let low = expert_act_half_q8g(DotBackend::scalar(), SHAPE, &slices, &x, 0..mid).unwrap();
        let high = expert_act_half_q8g(DotBackend::scalar(), SHAPE, &slices, &x, mid..SHAPE.inter)
            .unwrap();
        let mut act = low;
        act.extend_from_slice(&high);
        let mut qh = Q8gRow::zeros(SHAPE.inter).unwrap();
        let mut split = vec![0.0f32; SHAPE.hidden];
        expert_down_q8g(
            DotBackend::scalar(),
            SHAPE,
            &slices,
            &act,
            &mut qh,
            &mut split,
        )
        .unwrap();
        assert_eq!(
            whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            split.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
