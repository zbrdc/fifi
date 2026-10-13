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

/// On a host without AVX2 this prints "skipped" and passes, because then the scalar path is the only path
/// there is; on a host with it (the danlap and kubdan CPUs) it compares every bit.
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
