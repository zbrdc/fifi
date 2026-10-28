//! Zero, dead and subnormal input scales through the existing transform arithmetic.
//!
//! `project_f32` is a test-local mirror of the documented F32 path (scales widened exactly from
//! binary16, block Hadamards and the W' product in F32). It is checked against the repository's
//! fp64 reference before it is used; the fp64 reference itself is unchanged.

use std::{collections::HashMap, num::FpCategory};

use half::f16;

use super::fixture::MARKER;
use crate::{
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Config, Matrix},
    },
};

struct Components(HashMap<String, Tensor>);

impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

fn bits_of(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| f16::from_f32(*v).to_bits()).collect()
}

fn pseudo_words(seed: u64, count: usize) -> Vec<u16> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as u16
        })
        .collect()
}

/// Scales in [0.5, 1.5], binary16 bits.
fn ordinary(seed: u64, count: usize) -> Vec<u16> {
    let values: Vec<f32> = pseudo_words(seed, count)
        .iter()
        .map(|w| 0.5 + f32::from(*w) / 65_536.0)
        .collect();
    bits_of(&values)
}

/// Subnormal binary16 scales: bits 1..=0x3ff.
fn subnormals(seed: u64, count: usize) -> Vec<u16> {
    pseudo_words(seed, count)
        .iter()
        .map(|w| 1 + w % 0x3ff)
        .collect()
}

fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

fn half_tensor(words: &[u16]) -> Tensor {
    let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    tensor(Dtype::F16, vec![words.len() as u32], bytes)
}

fn matrix(suh: &[u16], svh: &[u16]) -> Matrix {
    let (k, n) = (suh.len(), svh.len());
    let trellis: Vec<u8> = pseudo_words(0x5eed, k / 16 * (n / 16) * 64)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let shape = vec![(k / 16) as u32, (n / 16) as u32, 64];
    let source = Components(HashMap::from([
        ("m.trellis".to_string(), tensor(Dtype::I16, shape, trellis)),
        ("m.suh".to_string(), half_tensor(suh)),
        ("m.svh".to_string(), half_tensor(svh)),
        (
            "m.mul1".to_string(),
            tensor(Dtype::I32, vec![], MARKER.to_le_bytes().to_vec()),
        ),
    ]));
    let config = Config::from_json(
        br#"{"quantization_config":{"quant_method":"exl3","version":"1.6.2","codebook":"mul1","bits":4.0,"head_bits":5,"expert_interm_div":128}}"#,
    )
    .unwrap();
    Matrix::load(&source, "m", &config).unwrap()
}

fn hadamard128(block: &mut [f32]) {
    let mut width = 1;
    while width < 128 {
        for start in (0..128).step_by(2 * width) {
            for lane in start..start + width {
                let (a, b) = (block[lane], block[lane + width]);
                block[lane] = a + b;
                block[lane + width] = a - b;
            }
        }
        width *= 2;
    }
    let scale = 1.0 / 128.0f32.sqrt();
    block.iter_mut().for_each(|v| *v *= scale);
}

fn widened(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
        .collect()
}

/// y = (((x * suh) H) W' H) * svh, every step in F32.
fn project_f32(matrix: &Matrix, x: &[f32]) -> Vec<f32> {
    let [k, n] = matrix.shape();
    let (suh, svh) = (widened(matrix.suh()), widened(matrix.svh()));
    let mut prime = vec![0u16; k * n];
    matrix
        .codebook()
        .reconstruct_into(matrix.trellis(), &mut prime)
        .unwrap();
    let mut z: Vec<f32> = x.iter().zip(&suh).map(|(x, s)| x * s).collect();
    for block in z.as_chunks_mut::<128>().0 {
        hadamard128(block);
    }
    let mut y = vec![0.0f32; n];
    for (row, z) in z.iter().enumerate() {
        for (col, y) in y.iter_mut().enumerate() {
            *y += z * f16::from_bits(prime[row * n + col]).to_f32();
        }
    }
    for block in y.as_chunks_mut::<128>().0 {
        hadamard128(block);
    }
    y.iter().zip(&svh).map(|(y, s)| y * s).collect()
}

fn project_f64(matrix: &Matrix, x: &[f32]) -> Vec<f64> {
    let input: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
    let mut output = vec![f64::NAN; matrix.shape()[1]];
    matrix.project_fp64_into(&input, 1, &mut output).unwrap();
    output
}

fn input(count: usize) -> Vec<f32> {
    pseudo_words(9, count)
        .iter()
        .map(|w| f32::from(*w) / 32_768.0 - 1.0)
        .collect()
}

fn relative_l2(actual: &[f32], expected: &[f64]) -> f64 {
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (f64::from(*a) - e).powi(2))
        .sum();
    (error / expected.iter().map(|e| e * e).sum::<f64>()).sqrt()
}

const N: usize = 128;

#[test]
fn the_f32_mirror_agrees_with_the_fp64_reference_at_ordinary_scales() {
    let m = matrix(&ordinary(1, N), &ordinary(2, 2 * N));
    let x = input(N);
    let reference = project_f64(&m, &x);
    assert!(reference.iter().any(|v| v.abs() > 1e-3));
    assert!(relative_l2(&project_f32(&m, &x), &reference) < 1e-5);
}

#[test]
fn zero_scale_vectors_and_dead_experts_give_exact_zeros_without_nan() {
    let zeros = vec![0u16; N];
    for (suh, svh) in [
        (zeros.clone(), ordinary(2, N)),
        (ordinary(1, N), zeros.clone()),
        (zeros.clone(), zeros.clone()),
    ] {
        let m = matrix(&suh, &svh);
        let x = input(N);
        assert!(project_f64(&m, &x).iter().all(|v| *v == 0.0));
        assert!(project_f32(&m, &x).iter().all(|v| *v == 0.0));
    }
}

#[test]
fn a_zero_scale_is_a_literal_zero_not_a_refit_or_epsilon() {
    let (mut suh, mut svh) = (ordinary(1, N), ordinary(2, N));
    let baseline = matrix(&suh, &svh);
    let x = input(N);
    let base32 = project_f32(&baseline, &x);
    let base64 = project_f64(&baseline, &x);
    // svh[7] = 0: exactly that output is zero and every other output is unchanged.
    svh[7] = 0;
    let m = matrix(&suh, &svh);
    let y32 = project_f32(&m, &x);
    let y64 = project_f64(&m, &x);
    assert_eq!((y32[7], y64[7]), (0.0, 0.0));
    for column in (0..N).filter(|c| *c != 7) {
        assert_eq!(y32[column].to_bits(), base32[column].to_bits());
        assert_eq!(y64[column].to_bits(), base64[column].to_bits());
    }
    // suh[5] = 0 removes input 5 exactly: the same result as zeroing that input.
    svh[7] = ordinary(2, N)[7];
    suh[5] = 0;
    let m = matrix(&suh, &svh);
    let mut without_input = x.clone();
    without_input[5] = 0.0;
    assert_eq!(
        project_f32(&m, &x)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        project_f32(&m, &without_input)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn subnormal_input_scales_survive_f32_scaling() {
    let suh = subnormals(3, N);
    assert!(
        suh.iter()
            .all(|b| f16::from_bits(*b).classify() == FpCategory::Subnormal)
    );
    let svh = ordinary(2, N);
    let m = matrix(&suh, &svh);
    let x = input(N);
    let reference = project_f64(&m, &x);
    let y = project_f32(&m, &x);
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(
        reference.iter().map(|v| v * v).sum::<f64>().sqrt() > 1e-6,
        "not flushed"
    );
    assert!(relative_l2(&y, &reference) < 1e-5);

    // Lifting every subnormal scale by 2^10 lands on exact binary16 normals; the outputs scale
    // by exactly 2^10 (to F32 rounding), so no information was lost below the normal range.
    let lifted: Vec<u16> = suh
        .iter()
        .map(|b| f16::from_f32(f16::from_bits(*b).to_f32() * 1024.0).to_bits())
        .collect();
    assert!(lifted.iter().all(|b| f16::from_bits(*b).is_normal()));
    let y_lifted = project_f32(&matrix(&lifted, &svh), &x);
    let rescaled: Vec<f64> = y_lifted.iter().map(|v| f64::from(*v) / 1024.0).collect();
    assert!(relative_l2(&y, &rescaled) < 1e-5);
}

#[test]
fn mixed_zero_subnormal_and_ordinary_scales_stay_finite_in_both_paths() {
    let mut suh = ordinary(1, N);
    let mut svh = ordinary(2, 2 * N);
    suh[..40].copy_from_slice(&subnormals(4, 40));
    suh[40..50].fill(0);
    svh[..20].copy_from_slice(&subnormals(5, 20));
    svh[20..30].fill(0);
    let m = matrix(&suh, &svh);
    let x = input(N);
    let reference = project_f64(&m, &x);
    let y = project_f32(&m, &x);
    assert!(reference.iter().all(|v| v.is_finite()));
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(relative_l2(&y, &reference) < 1e-5);
}
