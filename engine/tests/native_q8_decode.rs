//! Native tests of the fused 8-bit KV decode attention (packet FUSED-KV8-a):
//! `quantized_attention_decode` against the composite `quantized_attention_prefix` and the
//! dequantized `grouped_attention` oracle, the poisoned tail beyond `len`, the block
//! partition of `decode_blocks`, the decision 0032 fault hooks, and the peak-memory bound of
//! one call.

#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{sync::Mutex, time::Instant};

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, QuantizedKv},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

fn f32_input(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

/// A deterministic F32 tensor of `shape` from a value function, without naming every value.
fn f32_pattern(shape: &[u32], value: impl Fn(usize) -> f32) -> Tensor {
    let count: usize = shape.iter().map(|&dim| dim as usize).product();
    let values: Vec<f32> = (0..count).map(value).collect();
    f32_input(shape.to_vec(), &values)
}

/// A deterministic U32 tensor of `shape` from a value function: the code words of the
/// negative tests below.
fn u32_pattern(shape: &[u32], value: impl Fn(usize) -> u32) -> Tensor {
    let count: usize = shape.iter().map(|&dim| dim as usize).product();
    Tensor::try_from_host(
        Dtype::U32,
        shape.to_vec(),
        (0..count)
            .flat_map(|index| value(index).to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Decodes BF16 readback bytes to F32 (each BF16 value is the top half of an F32).
fn bf16_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

fn import(kernels: &mut MlxKernels, tensor: &Tensor) -> ExecutionTensor {
    kernels.import_tensor(tensor).unwrap()
}

fn read(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Tensor {
    kernels.readback(tensor).unwrap()
}

/// Decodes a float readback of `dtype` (F32 or BF16) to F32 values.
fn decode(dtype: Dtype, tensor: &Tensor) -> Vec<f32> {
    match dtype {
        Dtype::F32 => values_of(tensor),
        Dtype::Bf16 => bf16_values(tensor),
        other => panic!("a float readback of {other:?}"),
    }
}

/// Reads one decode result back and asserts that every value is finite: the graph is
/// lazy, so a Metal compile or dispatch failure of a "usable" adapter surfaces at the
/// readback, not at the call that built the graph.
fn assert_finite_readback(kernels: &mut MlxKernels, dtype: Dtype, got: &ExecutionTensor) {
    for (index, &value) in decode(dtype, &read(kernels, got)).iter().enumerate() {
        assert!(
            value.is_finite(),
            "element {index} of the valid call is {value}"
        );
    }
}

/// The affine 8-bit spec of every test here: group `group`, bits 8.
fn affine(group: u32) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits: 8,
        group_size: group,
    }
}

/// The first `len` rows (axis two) of a rank-four host tensor, head by head.
fn first_rows(tensor: &Tensor, len: u32) -> Tensor {
    let &[_, heads, cap, last] = tensor.shape() else {
        panic!(
            "a quantized buffer must be rank four, got {:?}",
            tensor.shape()
        )
    };
    // U32 codes and F32 scales and biases are four bytes each; a BF16 scale row is two.
    let width = match tensor.dtype() {
        Dtype::Bf16 => 2usize,
        _ => 4usize,
    };
    let bytes = tensor.host_bytes().unwrap();
    let span = last as usize * width;
    let mut kept = Vec::with_capacity(heads as usize * len as usize * span);
    for head in 0..heads as usize {
        let begin = head * cap as usize * span;
        kept.extend_from_slice(&bytes[begin..begin + len as usize * span]);
    }
    Tensor::try_from_host(tensor.dtype(), vec![1, heads, len, last], kept).unwrap()
}

/// Dequantizes the first `len` positions of one quantized capacity side through the existing
/// `dequantize` operation: an F32 `[1, KVH, len, D]` resident that `grouped_attention`
/// consumes as the independent oracle of the quantized prefix. Over stored BF16 scales the
/// shim casts the scales, so the F32 reconstruction stays exact.
fn dequantized_prefix(
    kernels: &mut MlxKernels,
    side: &QuantizedKv<'_>,
    len: u32,
    group: u32,
) -> ExecutionTensor {
    let heads = side.codes.shape()[1];
    let width = side.codes.shape()[3] * 4; // 8-bit codes: four values per U32 word.
    let codes = first_rows(&read(kernels, side.codes), len);
    let scales = first_rows(&read(kernels, side.scales), len);
    let biases = first_rows(&read(kernels, side.biases), len);
    let packed = PackedWeight::try_new(
        codes,
        scales,
        Some(biases),
        vec![1, heads, len, width],
        affine(group),
    )
    .unwrap();
    let resident = kernels.import_packed(&packed).unwrap();
    kernels.dequantize(&resident).unwrap()
}

/// One quantized capacity side whose scales and biases carry `dtype`: the F32 source of
/// `value` is imported and cast first, so `quantize_rows` returns scales and biases of `dtype`
/// beside the U32 codes - F32 stays itself, BF16 gives the BF16 cache of the BF16 cases.
fn quantized_capacity_of_dtype(
    kernels: &mut MlxKernels,
    kv_heads: u32,
    rows: u32,
    width: u32,
    group: u32,
    dtype: Dtype,
    value: impl Fn(usize) -> f32,
) -> (ExecutionTensor, ExecutionTensor, ExecutionTensor) {
    let mut source = import(kernels, &f32_pattern(&[1, kv_heads, rows, width], value));
    if dtype == Dtype::Bf16 {
        source = kernels.cast(&source, Dtype::Bf16).unwrap();
    }
    let (codes, scales, biases) = kernels.quantize_rows(&source, group, 8).unwrap();
    assert_eq!(codes.shape(), [1, kv_heads, rows, width / 4]);
    assert_eq!(scales.shape(), [1, kv_heads, rows, width / group]);
    assert_eq!(scales.dtype(), dtype);
    (codes, scales, biases)
}

/// One host copy of a quantized buffer whose rows `t >= len` carry poison: all-ones code
/// words for the U32 codes, and a NaN - or, for the zero-tail control, a zero - in the
/// buffer's own dtype for the scales and biases. Any read at or beyond `len` then surfaces
/// in the output: a NaN scale poisons every score of its row, and an all-ones code word
/// carries four 255 codes.
fn poisoned_buffer(
    kernels: &mut MlxKernels,
    tensor: &ExecutionTensor,
    len: u32,
    nan: bool,
) -> ExecutionTensor {
    let host = read(kernels, tensor);
    let &[_, heads, cap, last] = host.shape() else {
        panic!(
            "a quantized buffer must be rank four, got {:?}",
            host.shape()
        )
    };
    let dtype = host.dtype();
    let tail: Vec<u8> = match dtype {
        Dtype::U32 => vec![0xFF; last as usize * 4],
        Dtype::F32 => (0..last as usize)
            .flat_map(|_| (if nan { f32::NAN } else { 0.0 }).to_le_bytes())
            .collect(),
        Dtype::Bf16 => (0..last as usize)
            .flat_map(|_| (if nan { 0x7FC0u16 } else { 0x0000u16 }).to_le_bytes())
            .collect(),
        other => panic!("a quantized buffer of {other:?}"),
    };
    let byte_width = match dtype {
        Dtype::Bf16 => 2usize,
        _ => 4usize,
    };
    let span = last as usize * byte_width;
    assert_eq!(tail.len(), span, "the tail filler covers one whole row");
    let mut bytes = host.host_bytes().unwrap().to_vec();
    for head in 0..heads as usize {
        let base = head * cap as usize * span;
        for t in len as usize..cap as usize {
            let row = base + t * span;
            bytes[row..row + span].copy_from_slice(&tail);
        }
    }
    import(
        kernels,
        &Tensor::try_from_host(dtype, host.shape().to_vec(), bytes).unwrap(),
    )
}

/// The key values of one case: `k[t, d] = ((t' % 5 - 2) / 4) · ((d % 7 - 3) / 4)` with the
/// per-head phase `t' = t + 3 · head`, so |k| <= 0.375 with no long-range correlation and the
/// two KV heads hold DIFFERENT rows - a wrong head-to-KV mapping cannot match the oracle -
/// and - when `outliers` is set - every fourth row carries one quantization group at about
/// 6 units, an ~16x outlier.
fn key_value(width: u32, cap: u32, outliers: bool) -> impl Fn(usize) -> f32 {
    move |index| {
        let lane = index % width as usize;
        let head = (index / width as usize) / cap as usize;
        let t = (index / width as usize) % cap as usize;
        let base =
            ((((t + 3 * head) % 5) as f32) - 2.0) / 4.0 * ((((lane % 7) as f32) - 3.0) / 4.0);
        let group = if width >= 128 { 1 } else { 0 };
        if outliers && t % 4 == 1 && lane / 64 == group {
            base + 6.0 * (((lane % 3) as f32) - 1.0)
        } else {
            base
        }
    }
}

/// The value values of one case: `v[t, d] = ((t' % 9 - 4) / 4) · ((d % 11 - 5) / 4) +
/// 0.5 · (head + 1) · ((d % 5) - 2)` with the per-head phase `t' = t + 5 · head` (a
/// different multiplier than the keys). The first factor keeps a zero-mean, |v| <= 1.25
/// component; the second is a head-dependent OFFSET, constant in t, which gives the two KV
/// heads DIFFERENT MEANS of their values (head 0 offsets a lane by at most 1, head 1 by at
/// most 2). Without it the zero-mean factor makes the heads' attention outputs agree to
/// within the tolerance at long lengths - every full 45-row period sums to zero - which
/// would hide a wrong head-to-KV mapping; with it the outputs stay O(1) at every length.
/// The offset keeps every value a multiple of 1/16 below 8.25, so the BF16 casts of the
/// BF16 cases stay exact and the codes stay inside the 8-bit range. The keys carry their
/// own per-head phase (`t + 3 · head`), so the heads differ in the scores as well. When
/// `outliers` is set, every fourth row (a different phase than the key outliers) carries
/// one group at about 5 units: group `min(2, width / 64 - 1)`, so D 128, whose two groups
/// are 0 and 1, has value outliers too.
fn value_value(width: u32, cap: u32, outliers: bool) -> impl Fn(usize) -> f32 {
    move |index| {
        let lane = index % width as usize;
        let head = (index / width as usize) / cap as usize;
        let t = (index / width as usize) % cap as usize;
        let base = ((((t + 5 * head) % 9) as f32) - 4.0) / 4.0
            * ((((lane % 11) as f32) - 5.0) / 4.0)
            + 0.5 * (head as f32 + 1.0) * (((lane % 5) as f32) - 2.0);
        let group = (width as usize / 64 - 1).min(2);
        if outliers && t % 4 == 3 && lane / 64 == group {
            base + 5.0 * (((lane % 3) as f32) - 1.0)
        } else {
            base
        }
    }
}

/// The query values of one case: `q[h, d] = (d % 7 - 3) / 4`, |q| <= 0.75.
fn query_value() -> impl Fn(usize) -> f32 {
    move |index| (((index % 7) as f32) - 3.0) / 4.0
}

/// The F32 tolerance of the fused output against a reference (packet FUSED-KV8-a test a,
/// derived here): the fused kernel and the
/// reference see the SAME F32 dequantized values (the codes are integers and the scales and
/// biases widen to F32 exactly), so only the F32 accumulation order differs. The score dot
/// over D <= 256 lanes with partial sums bounded by about 8 wanders by
/// sqrt(256) · 2^-24 · 8 ≈ 8e-6; the softmax turns a score error eps into a probability
/// error of at most 2 · eps · p; the value mix over len <= 131,072 keys runs through the
/// online-softmax rescaling and the block merges, an effective depth of about
/// sqrt(len) · 2^-24 · |v| ≈ 362 · 6e-8 · 8.25 ≈ 1.8e-4 at the longest sweep (the outlier
/// rows reach |v| = 8.25 with the head offsets). The total stays under 2e-4, an order of
/// magnitude below the asserted bounds:
/// `F32_ORACLE_TOL` = 2e-3 against the dequantized oracle and `F32_COMPOSITE_TOL` = 5e-3
/// against the composite, whose `quantized_matmul` accumulates in yet another order, each as
/// atol and rtol (`tol · max(1, |reference|)`). A real failure - a wrong code byte order,
/// group index, block merge, or length overrun - moves outputs by O(0.1..1) or produces NaN,
/// far beyond both bounds.
const F32_ORACLE_TOL: f32 = 2e-3;
const F32_COMPOSITE_TOL: f32 = 5e-3;

/// The BF16 tolerance of the fused output against either reference: at most 4 units in the
/// last place of BF16 relative to the largest output magnitude of the row, plus the 2e-3 F32
/// floor for a row whose outputs cancel to near zero. The arithmetic: BF16 stores 7 mantissa
/// bits, so one ulp at magnitude M is at most 2^-7 · M, and 4 ulp is 4 · 2^-7 · M = 2^-5 · M
/// = M / 32 - about 0.031 for the rows whose outputs reach 1.0 (the head-offset lanes)
/// and 0.1 for the outlier rows (outputs up to about 3.3). The fused kernel rounds once at
/// its output cast, the oracle's `grouped_attention` rounds once
/// at its output, and the composite's BF16 scores and probabilities round one more handful of
/// times at 2^-8 relative each; every one of those stays under the same 4-ulp row bound
/// because the bound scales with the row maximum.
fn bf16_tolerance(row: &[f32]) -> f32 {
    let magnitude = row
        .iter()
        .fold(0.0f32, |largest, &value| largest.max(value.abs()));
    (4.0 * magnitude / 128.0).max(F32_ORACLE_TOL)
}

/// Runs one accuracy case (packet FUSED-KV8-a test a): the fused output against the composite
/// `quantized_attention_prefix` (L = 1, non-causal) and against `grouped_attention` over the
/// dequantized prefix on the same quantized buffers, printing the maximum observed error of
/// every comparison and, on a `q8-decode-gap` line, the observed KV-head gap with its
/// guard. Every compared value must be finite: a NaN never exceeds a tolerance bound, so
/// the explicit finiteness assert is what catches one.
fn check_decode_case(
    kernels: &mut MlxKernels,
    (heads, kv_heads, width): (u32, u32, u32),
    cap: u32,
    lens: &[u32],
    dtype: Dtype,
    outliers: bool,
) {
    if dtype == Dtype::Bf16 {
        kernels.allow_bf16(true);
    }
    let group = 64u32;
    let scale = 1.0 / (width as f32).sqrt();
    let mut query = import(kernels, &f32_pattern(&[1, heads, 1, width], query_value()));
    if dtype == Dtype::Bf16 {
        query = kernels.cast(&query, Dtype::Bf16).unwrap();
    }
    let (k_codes, k_scales, k_biases) = quantized_capacity_of_dtype(
        kernels,
        kv_heads,
        cap,
        width,
        group,
        dtype,
        key_value(width, cap, outliers),
    );
    let (v_codes, v_scales, v_biases) = quantized_capacity_of_dtype(
        kernels,
        kv_heads,
        cap,
        width,
        group,
        dtype,
        value_value(width, cap, outliers),
    );
    let keys = QuantizedKv {
        codes: &k_codes,
        scales: &k_scales,
        biases: &k_biases,
    };
    let values = QuantizedKv {
        codes: &v_codes,
        scales: &v_scales,
        biases: &v_biases,
    };
    let outlier_name = if outliers { "outliers" } else { "normal" };
    let dtype_name = if dtype == Dtype::Bf16 { "bf16" } else { "f32" };
    for &len in lens {
        let got = kernels
            .quantized_attention_decode(&query, &keys, &values, len, scale, group, 8)
            .unwrap();
        assert_eq!(
            got.shape(),
            [1, heads, 1, width],
            "heads {heads} over {kv_heads}, width {width}, {dtype_name}, {outlier_name}, \
             len {len}"
        );
        assert_eq!(got.dtype(), dtype, "the output carries the query dtype");
        let got_values = decode(dtype, &read(kernels, &got));

        // Reference 1: the composite on the same quantized buffers.
        let composite = kernels
            .quantized_attention_prefix(&query, &keys, &values, len, scale, false, group, 8)
            .unwrap();
        let composite_values = decode(dtype, &read(kernels, &composite));

        // Reference 2: grouped_attention over the dequantized prefix, the independent oracle
        // (F32 even for BF16 queries, because the dequantized prefix is F32 and MLX promotes).
        let oracle_k = dequantized_prefix(kernels, &keys, len, group);
        let oracle_v = dequantized_prefix(kernels, &values, len, group);
        let oracle = kernels
            .grouped_attention(&query, &oracle_k, &oracle_v, scale, false)
            .unwrap();
        let oracle_values = values_of(&read(kernels, &oracle));

        let base = format!("q{heads}k{kv_heads}d{width}-{dtype_name}-{outlier_name} len {len}");

        // The KV head groups must disagree past a guard derived from the data: the value
        // pattern carries a head-dependent offset (see `value_value`), so oracle outputs
        // that agree across groups mean a wrong head-to-KV mapping could hide inside the
        // tolerance (a kernel that reads KV head 0 for every query head would produce
        // exactly that agreement). The hand arithmetic of the gap, small geometry
        // (QH 2, KVH 2, D 64): q(d) = (d%7-3)/4 and k_h(t,d) = ((t+3h)%5-2)/4 · q(d), so
        // the score of row t is s_h(t) = (1/8) · Σ_{d<64} q(d)² · ((t+3h)%5-2)/4, and
        // Σ_{d<64} q(d)² = 9 · 28/16 + 9/16 = 16.3125 (nine periods of 7 plus d = 63),
        // giving s_h(t) = 0.509765625 · ((t+3h)%5 - 2): the softmax weight of a row
        // depends only on its class r = (t+3h)%5. With v_h(t,d) = b_h(t)·u(d) +
        // 0.5(h+1)·w(d), b_h(t) = ((t+5h)%9-4)/4, u(d) = (d%11-5)/4, w(d) = (d%5)-2, the
        // offset passes through the softmax exactly: out_h(d) = μ_h·u(d) + 0.5(h+1)·w(d)
        // with μ_h = Σ_t p_t·b_h(t). Every full 45-row cycle contributes zero to μ_h
        // (each class r mod 5 covers all nine residues of b_h once, 5 and 9 being
        // coprime), so μ_h is a boundary remainder: μ_0 = -0.0132 and μ_1 = -0.0063 at
        // len 100, μ_0 = -0.0015 and μ_1 = +0.00002 at len 1001 (softmax over the exact
        // scores above), so |μ_0| + |μ_1| <= 0.0266. At the lanes d % 5 ∈ {0, 4} the gap
        // is |(μ_0-μ_1)·u(d) ± 1| >= 1 - 1.25 · 0.0266 = 0.967, while the row-A magnitude
        // is at most 0.0166 + 1.0 + 1.25 = 2.27 (base, offset, and outlier mean), so the
        // guards are at most 2e-3 · 2.27 = 4.5e-3 (F32) and 4 · 2.27/128 = 0.071 (BF16):
        // the gap clears the guard by a factor of at least 13.6 at len 100, and len 1001
        // a fortiori (smaller μ). A scratch simulation of the same oracle math at the
        // long lengths 4,095, 4,096, 8,192, 32,768, and 131,072 gives gaps of 1.000 to
        // 1.004 against guards of at most 0.0703 (the BF16 outlier rows), a factor of at
        // least 14.2 at every length, dtype, and row kind.
        let rep = heads / kv_heads;
        for group in 1..kv_heads as usize {
            for j in 0..rep as usize {
                let head_a = j * width as usize;
                let head_b = (group * rep as usize + j) * width as usize;
                let row_a = &oracle_values[head_a..head_a + width as usize];
                let row_b = &oracle_values[head_b..head_b + width as usize];
                let magnitude = row_a
                    .iter()
                    .fold(0.0f32, |largest, &value| largest.max(value.abs()));
                let guard = if dtype == Dtype::Bf16 {
                    bf16_tolerance(row_a)
                } else {
                    F32_ORACLE_TOL * magnitude.max(1.0)
                };
                let gap = row_a
                    .iter()
                    .zip(row_b)
                    .fold(0.0f32, |gap, (&a, &b)| gap.max((a - b).abs()));
                println!(
                    "q8-decode-gap {base} group {group} head {j}: gap {gap:.6}, guard {guard:.6}"
                );
                assert!(
                    gap > guard,
                    "{base} KV head groups 0 and {group} agree to {gap}, inside the tolerance \
                     {guard}: a wrong head mapping could hide"
                );
            }
        }

        let mut worst_composite = 0.0f32;
        let mut worst_oracle = 0.0f32;
        let mut largest = 0.0f32;
        for head in 0..heads as usize {
            let row = head * width as usize;
            let oracle_row = &oracle_values[row..row + width as usize];
            let row_tol = if dtype == Dtype::Bf16 {
                bf16_tolerance(oracle_row)
            } else {
                F32_ORACLE_TOL
            };
            for (lane, ((&actual, &oracle_value), &composite_value)) in got_values
                [row..row + width as usize]
                .iter()
                .zip(oracle_row)
                .zip(&composite_values[row..row + width as usize])
                .enumerate()
            {
                assert!(
                    actual.is_finite() && oracle_value.is_finite() && composite_value.is_finite(),
                    "{base} head {head} lane {lane}: a nonfinite value (got {actual}, oracle \
                     {oracle_value}, composite {composite_value})"
                );
                // The F32 bounds are atol and rtol in one (`tol · max(1, |reference|)`); the
                // BF16 bound is the absolute 4-ulp row constant.
                let oracle_bound = if dtype == Dtype::Bf16 {
                    row_tol
                } else {
                    row_tol * oracle_value.abs().max(1.0)
                };
                let composite_bound = if dtype == Dtype::Bf16 {
                    row_tol
                } else {
                    F32_COMPOSITE_TOL * composite_value.abs().max(1.0)
                };
                let oracle_error = (actual - oracle_value).abs();
                let composite_error = (actual - composite_value).abs();
                worst_oracle = worst_oracle.max(oracle_error);
                worst_composite = worst_composite.max(composite_error);
                largest = largest
                    .max(actual.abs())
                    .max(oracle_value.abs())
                    .max(composite_value.abs());
                assert!(
                    oracle_error <= oracle_bound,
                    "{base} head {head} lane {lane}: fused {actual} against the dequantized \
                     oracle {oracle_value} (bound {oracle_bound})"
                );
                assert!(
                    composite_error <= composite_bound,
                    "{base} head {head} lane {lane}: fused {actual} against the composite \
                     {composite_value} (bound {composite_bound})"
                );
            }
        }
        println!(
            "q8-decode-error {base}: max |fused - oracle| {worst_oracle:.3e}, max |fused - \
             composite| {worst_composite:.3e}, largest output magnitude {largest:.3e}"
        );
    }
}

/// The fused output against BOTH references on the same quantized buffers (packet
/// FUSED-KV8-a test a): the composite `quantized_attention_prefix` with `L = 1` and
/// non-causal, and `grouped_attention` over the dequantized prefix as the independent
/// oracle. Configs QH/KVH/D = 16/2/256 (the real model, REP 8) and 4/2/64 (the test shape,
/// REP 2), group 64, lengths 1, 2, 63, 64, 65, 255, 256, 257, 4,095, 4,096, and 32,768 at
/// capacity 32,768, F32 and BF16 queries, normal rows and rows with one outlier group -
/// every comparison runs both, because only the outlier rows make a wrong bias or scale
/// visible - plus one capacity-131,072 run at len 131,072 in each dtype with both row kinds
/// (the deep run is where the 256/128-block tier applies). The tolerances are derived in
/// the docs of [`F32_ORACLE_TOL`] and [`bf16_tolerance`], and every comparison prints its
/// observed maximum error and largest output magnitude on one `q8-decode-error` line.
#[test]
fn quantized_attention_decode_matches_the_composite_and_the_dequantized_oracle() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let cap = 32768u32;
    let lens = &[1u32, 2, 63, 64, 65, 255, 256, 257, 4095, 4096, 32768];
    {
        let mut kernels = MlxKernels::try_new().unwrap();
        for &(heads, kv_heads, width) in &[(16u32, 2u32, 256u32), (4u32, 2u32, 64u32)] {
            for &dtype in &[Dtype::F32, Dtype::Bf16] {
                for &outliers in &[false, true] {
                    check_decode_case(
                        &mut kernels,
                        (heads, kv_heads, width),
                        cap,
                        lens,
                        dtype,
                        outliers,
                    );
                }
            }
        }
    }
    // The one deep-context run: capacity 131,072 at full length, both dtypes, normal AND
    // outlier rows - the deep run is where the 256/128-block tier applies, and only
    // outlier rows make a wrong bias or scale visible there.
    let deep_cap = 131072u32;
    let mut kernels = MlxKernels::try_new().unwrap();
    for &(heads, kv_heads, width) in &[(16u32, 2u32, 256u32), (4u32, 2u32, 64u32)] {
        for &dtype in &[Dtype::F32, Dtype::Bf16] {
            for &outliers in &[false, true] {
                check_decode_case(
                    &mut kernels,
                    (heads, kv_heads, width),
                    deep_cap,
                    &[deep_cap],
                    dtype,
                    outliers,
                );
            }
        }
    }
}

/// The buffer tail beyond `len` never feeds the output (packet FUSED-KV8-a test b): with
/// all-ones codes and NaN scales and biases at every position `t >= len`, the output stays
/// finite and equals, value for value, the run over buffers whose tail is zero - the kernel
/// reads only `t < len`. Two geometries: the small one (QH 4 over KVH 2, D 64) at len 1001
/// of capacity 1024, and the real-model one (QH 16 over KVH 2, D 256, REP 8) at len 1001
/// AND len 1000 - the block size is 8 and divides 1000 but not 1001, so a partial final
/// block and empty tail blocks both occur - in F32 and BF16. A read past `len` would pull a
/// NaN scale into a score and surface as NaN or a shifted value, far beyond the equality.
#[test]
fn quantized_attention_decode_never_reads_beyond_len() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let group = 64u32;
    for (heads, kv_heads, width, cap, lens) in [
        (4u32, 2u32, 64u32, 1024u32, &[1001u32][..]),
        (16u32, 2u32, 256u32, 1024u32, &[1001u32, 1000][..]),
    ] {
        let scale = 1.0 / (width as f32).sqrt();
        for &dtype in &[Dtype::F32, Dtype::Bf16] {
            let mut kernels = MlxKernels::try_new().unwrap();
            if dtype == Dtype::Bf16 {
                kernels.allow_bf16(true);
            }
            let keys = quantized_capacity_of_dtype(
                &mut kernels,
                kv_heads,
                cap,
                width,
                group,
                dtype,
                key_value(width, cap, false),
            );
            let values = quantized_capacity_of_dtype(
                &mut kernels,
                kv_heads,
                cap,
                width,
                group,
                dtype,
                value_value(width, cap, false),
            );
            let mut query = import(
                &mut kernels,
                &f32_pattern(&[1, heads, 1, width], query_value()),
            );
            if dtype == Dtype::Bf16 {
                query = kernels.cast(&query, Dtype::Bf16).unwrap();
            }
            for &len in lens {
                // The poisoned pair: NaN scales and biases, all-ones codes beyond `len`.
                let poisoned_keys = (
                    poisoned_buffer(&mut kernels, &keys.0, len, true),
                    poisoned_buffer(&mut kernels, &keys.1, len, true),
                    poisoned_buffer(&mut kernels, &keys.2, len, true),
                );
                let poisoned_values = (
                    poisoned_buffer(&mut kernels, &values.0, len, true),
                    poisoned_buffer(&mut kernels, &values.1, len, true),
                    poisoned_buffer(&mut kernels, &values.2, len, true),
                );
                /// The `QuantizedKv` view of one (codes, scales, biases) triple.
                fn kv<'a>(
                    side: &'a (ExecutionTensor, ExecutionTensor, ExecutionTensor),
                ) -> QuantizedKv<'a> {
                    QuantizedKv {
                        codes: &side.0,
                        scales: &side.1,
                        biases: &side.2,
                    }
                }
                let got = kernels
                    .quantized_attention_decode(
                        &query,
                        &kv(&poisoned_keys),
                        &kv(&poisoned_values),
                        len,
                        scale,
                        group,
                        8,
                    )
                    .unwrap();
                let got_values = decode(dtype, &read(&mut kernels, &got));
                for (index, &value) in got_values.iter().enumerate() {
                    assert!(
                        value.is_finite(),
                        "a read past len {len} surfaced: element {index} is {value}"
                    );
                }

                // The zero-tail control over the same first `len` rows: the two runs compute
                // the identical F32 sums in the identical launch shape, so the outputs must
                // match value for value.
                let zero_keys = (
                    poisoned_buffer(&mut kernels, &keys.0, len, false),
                    poisoned_buffer(&mut kernels, &keys.1, len, false),
                    poisoned_buffer(&mut kernels, &keys.2, len, false),
                );
                let zero_values = (
                    poisoned_buffer(&mut kernels, &values.0, len, false),
                    poisoned_buffer(&mut kernels, &values.1, len, false),
                    poisoned_buffer(&mut kernels, &values.2, len, false),
                );
                let control = kernels
                    .quantized_attention_decode(
                        &query,
                        &kv(&zero_keys),
                        &kv(&zero_values),
                        len,
                        scale,
                        group,
                        8,
                    )
                    .unwrap();
                let control_values = decode(dtype, &read(&mut kernels, &control));
                let dtype_name = if dtype == Dtype::Bf16 { "bf16" } else { "f32" };
                for (index, (a, b)) in got_values.iter().zip(&control_values).enumerate() {
                    assert!(
                        a == b,
                        "{dtype_name} element {index}: the poisoned tail changed the output ({a} vs {b})"
                    );
                }
                println!(
                    "{dtype_name} q{heads}k{kv_heads}d{width} len {len} of {cap}: the poisoned-tail \
                     output equals the zero-tail output over {} elements",
                    got_values.len()
                );
            }
        }
    }
}

/// The block partition (packet FUSED-KV8-a test c): lengths that straddle the thresholds of
/// `decode_blocks` - 8,192 (the last 64-block length), 8,193 and 32,768 (the 128-block tier
/// for REP 8, 64 blocks for REP 2), and 32,769 (the 256-block tier, 128 for REP 2) at
/// capacity 33,024, plus len 5 where `NB = len` for REP 8 - against the dequantized oracle,
/// so every block-count branch and the merge run for both REP 8 (QH 16 over KVH 2, D 256)
/// and REP 2 (4 over 2, D 64), with normal AND outlier
/// rows: only the outlier rows make a wrong bias or scale visible at these block tiers. The
/// tolerance is the derived F32 oracle bound of test a.
#[test]
fn quantized_attention_decode_partitions_straddle_the_block_thresholds() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (cap, lens) = (33024u32, &[8192u32, 8193, 32768, 32769]);
    let mut kernels = MlxKernels::try_new().unwrap();
    for &(heads, kv_heads, width) in &[(16u32, 2u32, 256u32), (4u32, 2u32, 64u32)] {
        for &outliers in &[false, true] {
            check_decode_case(
                &mut kernels,
                (heads, kv_heads, width),
                cap,
                lens,
                Dtype::F32,
                outliers,
            );
        }
    }
    // NB = len: five blocks of one key each at REP 8, both row kinds.
    for &outliers in &[false, true] {
        check_decode_case(&mut kernels, (16, 2, 256), 1024, &[5], Dtype::F32, outliers);
    }
}

/// The small-repeat geometries (packet FUSED-KV8-a): (QH, KVH, D) = (2, 2, 64)
/// with REP 1, (8, 2, 128) with REP 4, and (6, 2, 64) with REP 3, at len 100 and 1001 of
/// capacity 1024 - both a length the block size divides (100 = 12 blocks of 8 plus a block
/// of 4) and one it does not (1001) - F32 and BF16 queries, normal and outlier rows, against
/// the dequantized oracle (and the composite): the repeat range the shape check admits below
/// the REP 8 of the real model, including the D 128 head dim.
#[test]
fn quantized_attention_decode_covers_the_small_repeat_geometries() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut kernels = MlxKernels::try_new().unwrap();
    for &(heads, kv_heads, width) in &[
        (2u32, 2u32, 64u32),
        (8u32, 2u32, 128u32),
        (6u32, 2u32, 64u32),
    ] {
        for &dtype in &[Dtype::F32, Dtype::Bf16] {
            for &outliers in &[false, true] {
                check_decode_case(
                    &mut kernels,
                    (heads, kv_heads, width),
                    1024,
                    &[100, 1001],
                    dtype,
                    outliers,
                );
            }
        }
    }
}

/// Every repeat of the layout under the R4 partition (packet FUSED-KV8-EFF-2 test a):
/// REP 1 through 8 - here QH = 2 · REP over KVH 2 at D 64 - at the lengths 1 (one block),
/// 130 (NB 64 with twenty EMPTY tail blocks), 8,192 (the last 64-block length), and
/// 32,768 (the 128-block tier), against the composite `quantized_attention_prefix` and
/// the dequantized oracle with the tolerances of test a of FUSED-KV8-a, F32 and BF16
/// queries, normal and outlier rows. The partition differs per repeat - RPH 4 with SGS 2
/// at REP 8, RPH 3 with SGS 2 at REP 6, RPH = REP with SGS 1 at REP 1, 2, 3, and 4, RPH 1
/// with SGS 5 and 7 at the primes - so a head range, a simdgroup stride, or a launch row
/// that is wrong at exactly one repeat shows here. The sweep runs at D 64 (one quantization
/// group per row); test a covers D 256 at REP 8 and REP 2, and the group indexing of a
/// lane does not depend on the simdgroup, so the other repeats are not repeated at D 256.
#[test]
fn quantized_attention_decode_covers_every_repeat_of_the_r4_partition() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (cap, lens) = (32768u32, &[1u32, 130, 8192, 32768]);
    let mut kernels = MlxKernels::try_new().unwrap();
    for rep in 1u32..=8 {
        for &dtype in &[Dtype::F32, Dtype::Bf16] {
            for &outliers in &[false, true] {
                check_decode_case(&mut kernels, (2 * rep, 2, 64), cap, lens, dtype, outliers);
            }
        }
    }
}

/// The deep tier of the R4 partition on the real shape (packet FUSED-KV8-EFF-2 test b):
/// REP 8 at 131,072 keys with QH 16 over KVH 2 at D 256, the tier where `decode_blocks`
/// gives NB 256 and pass 1 runs 512 threadgroups (256 block rows over 2 KV heads) of two
/// simdgroups over the same blocks, against the composite and the dequantized oracle, F32
/// and BF16 queries, normal and outlier rows, the tolerances of test a of FUSED-KV8-a.
#[test]
fn quantized_attention_decode_rep8_deep_context_matches_the_references() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let deep_cap = 131072u32;
    let mut kernels = MlxKernels::try_new().unwrap();
    for &dtype in &[Dtype::F32, Dtype::Bf16] {
        for &outliers in &[false, true] {
            check_decode_case(
                &mut kernels,
                (16, 2, 256),
                deep_cap,
                &[deep_cap],
                dtype,
                outliers,
            );
        }
    }
}

/// The empty tail blocks of the new partition (packet FUSED-KV8-EFF-2 test c): len 130
/// gives NB 64 and a block size of 3, so blocks 44 to 63 hold no key and pass 1 writes
/// them the EMPTY partial - max -FLT_MAX, sum 0, zeros - that pass 2 must weight by
/// `exp(-FLT_MAX - m*) = 0`. Both partitions at once: REP 8 on the real shape (RPH 4,
/// SGS 2, D 256) and REP 4 (RPH 4, SGS 1, D 64), F32 and BF16 queries, normal and outlier
/// rows, against both references at capacity 1024; the poisoned-tail test covers the same
/// rule at len 1001.
#[test]
fn quantized_attention_decode_merges_the_empty_tail_blocks() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut kernels = MlxKernels::try_new().unwrap();
    for &(heads, kv_heads, width) in &[(16u32, 2u32, 256u32), (8u32, 2u32, 64u32)] {
        for &dtype in &[Dtype::F32, Dtype::Bf16] {
            for &outliers in &[false, true] {
                check_decode_case(
                    &mut kernels,
                    (heads, kv_heads, width),
                    1024,
                    &[130],
                    dtype,
                    outliers,
                );
            }
        }
    }
}

/// The fused decode survives raw attention scores far past the F32 exp overflow bound: the
/// online softmax of pass 1 and the log-sum-exp merge of pass 2 subtract the running and
/// the global max before every exp, so no exponent ever sees a score magnitude. The
/// arithmetic that reaches raw scores of up to about 353: the query is the constant 1.0
/// over all 256 lanes, `scale` is 1.0, every key row `t` holds the constant
/// `0.5 + (t % 4) / 8` plus a position ramp `ramp(t) = 0.5 · (1 - |2t/32,769 - 1|)` (a
/// triangle over the capacity: zero at both ends, one half at the middle row 16,384) plus
/// the per-lane term `(d % 8) / 1000`, and the raw score of row `t` is
///
/// ```text
/// sum_d q·k = 256 · (0.5 + (t % 4) / 8 + ramp(t)) + 0.001 · sum_{d<256} (d % 8)
///           = 256 · (0.5 + (t % 4) / 8 + ramp(t)) + 0.896   in [128.9, 352.9]
/// ```
///
/// exp overflows F32 near 88.7, so every score is far past the limit and any unshifted
/// exponent produces inf (and then NaN in the merge). The per-lane term keeps every
/// 64-wide quantization group off a constant (its values span 0.007) and the ramp is
/// constant across the lanes of a row, so the affine 8-bit error is at most half a step of
/// that span per value and the dequantized scores stay within 256 · 0.007 / 510 ≈ 0.0035
/// of these numbers - the assertion below checks on the host that every fed score stays at
/// or above 120, before the comparison runs. Lengths 8,193 and 32,769 at capacity 32,769
/// straddle the 128- and 256-block tiers of `decode_blocks` (REP 8), and the ramp makes the
/// BLOCK MAXIMA drift so the merge of pass 2 runs with unequal weights: at len 8,193 (127 blocks
/// of 65 rows, the last holding three, plus one empty block: 128 in all) the maxima rise from
/// 225.4 (block 0, whose best row is its last residue-3 row 63) to 288.9 (block 126, rows 8,190
/// to 8,192, best row 8,191); at len 32,769 (255 blocks of 129 rows, the last holding three, plus
/// one empty block: 256 in all) they rise from 225.9 (block 0, row 127) to 352.9 (block 127,
/// the middle block that holds the apex row 16,383) and fall to 224.9 (block 254, rows 32,766
/// to 32,768, best row 32,767). Adjacent blocks differ by about 0.5 (len 8,193) and 1.0
/// (len 32,769) score units; the empty block weighs `exp(-FLT_MAX - m*) = 0`.
///
/// The values carry a quadratic position drift `0.25 · ((2t - 32,769) / 32,769)² ·
/// ((d % 11 - 5) / 4)`: zero at the apex rows that dominate the correct merge, up to about
/// 0.31 at the ends. The equal-weight merge mutant (merge weight 1.0 instead of
/// `exp(pmax_b - mstar)`) averages every block's partial instead, and the mean of
/// ((2t - 32,769) / 32,769)² over uniformly spread blocks is 1/3 (the mean square of a
/// uniform over [-1, 1]), so it shifts the output by about 0.25/3 · 1.25 ≈ 0.104 at the
/// widest lanes - roughly fifty times the 2e-3 oracle bound - which is why the mutation
/// manifest maps that mutant to this test as well as to the partition test. With the
/// correct kernel the merge exponents `pmax_b - mstar` lie in [-128, 0] at len 32,769 and
/// [-63.5, 0] at len 8,193. The oracle is `grouped_attention` over the dequantized prefix -
/// MLX fast SDPA subtracts the row max, so it stays stable at these magnitudes - and every
/// compared value must be finite (a NaN never exceeds a tolerance bound) and inside the
/// derived F32 oracle tolerance. The `exp(pmax_b)` merge mutant (no `- mstar`) overflows
/// to inf here and fails the finiteness assert.
#[test]
fn quantized_attention_decode_survives_large_scores() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, kv_heads, width, group, cap) = (16u32, 2u32, 256u32, 64u32, 32769u32);
    let scale = 1.0f32;
    let large_key = |index: usize| {
        let lane = index % width as usize;
        let t = (index / width as usize) % cap as usize;
        let ramp = 0.5 * (1.0 - (2.0 * t as f32 / cap as f32 - 1.0).abs());
        0.5 + (t % 4) as f32 / 8.0 + ramp + (lane % 8) as f32 / 1000.0
    };
    let large_value = |index: usize| {
        let lane = index % width as usize;
        let t = (index / width as usize) % cap as usize;
        let drift = 0.25 * ((2.0 * t as f32 - cap as f32) / cap as f32).powi(2);
        ((((t % 9) as f32) - 4.0) / 4.0 + drift) * ((((lane % 11) as f32) - 5.0) / 4.0)
    };
    // The raw scores of the fed values, computed on the host: every row of the constant-1
    // query over the key rows of KV head 0 must stay at or above 120 (the smallest is
    // 128.9) and the ramp must carry the best row near its apex (the peak score is 352.9).
    let mut raw_min = f32::INFINITY;
    let mut raw_max = f32::NEG_INFINITY;
    for t in 0..cap as usize {
        let row: f32 = (0..width as usize)
            .map(|lane| large_key(t * width as usize + lane))
            .sum();
        raw_min = raw_min.min(row);
        raw_max = raw_max.max(row);
    }
    assert!(
        raw_min >= 120.0 && raw_max >= 300.0,
        "the raw scores must fill [120, 300], they span [{raw_min}, {raw_max}]"
    );

    let mut kernels = MlxKernels::try_new().unwrap();
    let query = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, width], |_| 1.0f32),
    );
    let keys = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        large_key,
    );
    let values = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        large_value,
    );
    let quantized_keys = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let quantized_values = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    for &len in &[8193u32, 32769] {
        let got = kernels
            .quantized_attention_decode(
                &query,
                &quantized_keys,
                &quantized_values,
                len,
                scale,
                group,
                8,
            )
            .unwrap();
        assert_eq!(got.shape(), [1, heads, 1, width]);
        let got_values = values_of(&read(&mut kernels, &got));

        let oracle_k = dequantized_prefix(&mut kernels, &quantized_keys, len, group);
        let oracle_v = dequantized_prefix(&mut kernels, &quantized_values, len, group);
        let oracle = kernels
            .grouped_attention(&query, &oracle_k, &oracle_v, scale, false)
            .unwrap();
        let oracle_values = values_of(&read(&mut kernels, &oracle));

        let mut worst = 0.0f32;
        let mut largest = 0.0f32;
        for head in 0..heads as usize {
            let row = head * width as usize;
            for (lane, (&actual, &oracle_value)) in got_values[row..row + width as usize]
                .iter()
                .zip(&oracle_values[row..row + width as usize])
                .enumerate()
            {
                assert!(
                    actual.is_finite() && oracle_value.is_finite(),
                    "len {len} head {head} lane {lane}: a nonfinite value under large scores \
                     (fused {actual}, oracle {oracle_value})"
                );
                let bound = F32_ORACLE_TOL * oracle_value.abs().max(1.0);
                let error = (actual - oracle_value).abs();
                worst = worst.max(error);
                largest = largest.max(actual.abs()).max(oracle_value.abs());
                assert!(
                    error <= bound,
                    "len {len} head {head} lane {lane}: fused {actual} against the dequantized \
                     oracle {oracle_value} (bound {bound})"
                );
            }
        }
        println!(
            "q8-decode-error large scores len {len}: max |fused - oracle| {worst:.3e}, \
             largest output magnitude {largest:.3e}"
        );
    }
}

/// The peak growth of one call (packet FUSED-KV8-a test e, at the block tier of packet
/// FUSED-KV8-EFF-2): at capacity 32,768 and len 32,768 with QH 16 over KVH 2 and D 256,
/// the block count is 128 and the call's scratch is the F32 partials `po` [16, 128, 256]
/// = 2,097,152 bytes plus `psum` and `pmax` [16, 128] = 8,192 bytes each, 2,113,536 bytes
/// in all, and the output [1, 16, 1, 256] is 16,384 bytes. The asserted bound is the
/// partials' total plus 1 MiB = 3,162,112 bytes: the
/// call reads the quantized buffers in place (no F32 prefix is built) and copies nothing
/// else, so a growth past the bound would mean a hidden copy of the prefix or of the
/// partials. The six buffers and the query are evaluated before the measurement, so the
/// measured call does not also materialize the quantization (the lesson of the KV8-OPS
/// view-cost test).
#[test]
fn quantized_attention_decode_peak_growth_stays_below_the_partials_plus_1mib() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, kv_heads, width, group, cap, len) =
        (16u32, 2u32, 256u32, 64u32, 32768u32, 32768u32);
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    let keys = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        key_value(width, cap, false),
    );
    let values = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        value_value(width, cap, false),
    );
    let query = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, width], query_value()),
    );
    let quantized_keys = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let quantized_values = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    // The buffers and the query are LAZY until the first evaluation; without this evaluation
    // the measured call would also materialize the quantization of the whole cache.
    kernels
        .evaluate(&[
            &query, &keys.0, &keys.1, &keys.2, &values.0, &values.1, &values.2,
        ])
        .unwrap();

    let blocks = 128u64; // decode_blocks(32768, 8): len > 8192, <= 32768, REP >= 6.
    let partials: u64 = u64::from(heads) * blocks * (u64::from(width) + 2) * 4;
    let output_bytes: u64 = u64::from(heads) * u64::from(width) * 4;
    let bound = partials + 1024 * 1024;
    assert_eq!(
        partials, 2_113_536,
        "the derived partial size: 16 · 128 · (256 + 2) F32 values"
    );

    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let began = Instant::now();
    let got = kernels
        .quantized_attention_decode(
            &query,
            &quantized_keys,
            &quantized_values,
            len,
            scale,
            group,
            8,
        )
        .unwrap();
    kernels.readback(&got).unwrap();
    let elapsed = began.elapsed().as_secs_f64() * 1e3;
    drop(got);
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(base);
    println!(
        "one fused decode at len {len}: peak growth {growth} bytes in {elapsed:.1} ms; the \
         F32 partials are {partials} bytes, the output {output_bytes}, the bound {bound}"
    );
    assert!(
        growth < bound,
        "the fused decode grew the peak by {growth} bytes, above the partials ({partials}) \
         plus 1 MiB - the call must not copy the prefix or the partials"
    );
}

/// The operands of the negative tests below (packet FUSED-KV8-a): a small pair
/// of capacity 64 at width 64, F32 scales, head-dependent rows, and an F32 query.
fn negative_case(
    kernels: &mut MlxKernels,
) -> (
    ExecutionTensor,
    (ExecutionTensor, ExecutionTensor, ExecutionTensor),
    (ExecutionTensor, ExecutionTensor, ExecutionTensor),
) {
    let (heads, kv_heads, width, group, cap) = (4u32, 2u32, 64u32, 64u32, 64u32);
    let query = import(kernels, &f32_pattern(&[1, heads, 1, width], query_value()));
    let keys = quantized_capacity_of_dtype(
        kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        key_value(width, cap, false),
    );
    let values = quantized_capacity_of_dtype(
        kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        value_value(width, cap, false),
    );
    (query, keys, values)
}

/// An operand that belongs to another adapter is refused before any native call (packet
/// FUSED-KV8-a), and the refusal leaves the adapter usable: a valid decode on the
/// same adapter right after it succeeds.
#[test]
fn quantized_attention_decode_refuses_an_operand_of_another_adapter() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, kv_heads, width, group, cap, len) = (4u32, 2u32, 64u32, 64u32, 64u32, 32u32);
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let (query, keys, values) = negative_case(&mut kernels);
    // The codes shape of one side, imported by ANOTHER adapter: `check_owner` refuses it
    // before any native call.
    let foreign = other
        .import_tensor(&u32_pattern(&[1, kv_heads, cap, width / 4], |index| {
            index as u32
        }))
        .unwrap();
    let mixed_keys = QuantizedKv {
        codes: &foreign,
        scales: &keys.1,
        biases: &keys.2,
    };
    let values_view = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    let refused =
        kernels.quantized_attention_decode(&query, &mixed_keys, &values_view, len, scale, group, 8);
    assert!(
        matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("another native owner")),
        "a foreign operand must be BadRequest: {:?}",
        refused.err()
    );
    // The adapter stays usable: the same call over its own codes succeeds, and the
    // readback proves the kernel compiled and ran - every value is finite.
    let keys_view = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let got = kernels
        .quantized_attention_decode(&query, &keys_view, &values_view, len, scale, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_finite_readback(&mut kernels, Dtype::F32, &got);
}

/// BF16 queries are refused while the adapter has not enabled `allow_bf16` (packet
/// FUSED-KV8-a), and the refusal leaves the adapter usable: after `allow_bf16`
/// the SAME call succeeds and returns BF16.
#[test]
fn quantized_attention_decode_refuses_bf16_queries_until_allow_bf16() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, _kv_heads, width, group) = (4u32, 2u32, 64u32, 64u32);
    let len = 32u32;
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);
    let (f32_query, keys, values) = negative_case(&mut kernels);
    let bf16_query = kernels.cast(&f32_query, Dtype::Bf16).unwrap();
    // Without BF16 operands the dtype rule refuses the queries before any native call.
    kernels.allow_bf16(false);
    let keys_view = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let values_view = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    let refused = kernels.quantized_attention_decode(
        &bf16_query,
        &keys_view,
        &values_view,
        len,
        scale,
        group,
        8,
    );
    assert!(
        matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("dtype")),
        "BF16 queries without allow_bf16 must be BadRequest: {:?}",
        refused.err()
    );
    // The adapter stays usable: after allow_bf16 the same call succeeds.
    kernels.allow_bf16(true);
    let got = kernels
        .quantized_attention_decode(&bf16_query, &keys_view, &values_view, len, scale, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_eq!(
        got.dtype(),
        Dtype::Bf16,
        "the output carries the query dtype"
    );
    assert_finite_readback(&mut kernels, Dtype::Bf16, &got);
}

/// `supports_quantized_attention_decode` says what the operation accepts, with no native call
/// and no state change (packet FUSED-KV8-b): the decoder chooses between the fused kernel and
/// the composite by this predicate, so it must agree with the operation on every shape it can
/// be asked about. The supported geometry (4 query heads over 2 KV heads, D 64, group 64, 8
/// bits, `len` inside the capacity) gives true and the operation succeeds; a length of 0, a
/// length above the capacity, bits 4, and group 32 give false and the operation refuses with
/// `BadRequest`; a D of 96 (not a multiple of 64) gives false.
#[test]
fn supports_quantized_attention_decode_agrees_with_the_operation() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, width, group, cap) = (4u32, 64u32, 64u32, 64u32);
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    let (query, keys, values) = negative_case(&mut kernels);
    let keys_view = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let values_view = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    assert!(kernels.supports_quantized_attention_decode(
        &query,
        &keys_view,
        &values_view,
        32,
        group,
        8
    ));
    for (len, group_size, bits, why) in [
        (0u32, group, 8u32, "length 0"),
        (cap + 1, group, 8, "length above the capacity"),
        (32, group, 4, "4 bits"),
        (32, 32, 8, "group 32"),
    ] {
        assert!(
            !kernels.supports_quantized_attention_decode(
                &query,
                &keys_view,
                &values_view,
                len,
                group_size,
                bits
            ),
            "the predicate must refuse {why}"
        );
        let refused = kernels.quantized_attention_decode(
            &query,
            &keys_view,
            &values_view,
            len,
            scale,
            group_size,
            bits,
        );
        assert!(
            matches!(&refused, Err(EngError::BadRequest(_))),
            "the operation must refuse {why} with BadRequest: {:?}",
            refused.err()
        );
    }
    // A head dimension of 96 is not a multiple of 64 (the lane slice rule): false.
    let wide_query = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, 96], query_value()),
    );
    assert!(!kernels.supports_quantized_attention_decode(
        &wide_query,
        &keys_view,
        &values_view,
        32,
        group,
        8
    ));
    // The predicate changed nothing: the supported call still succeeds on the same adapter.
    let got = kernels
        .quantized_attention_decode(&query, &keys_view, &values_view, 32, scale, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_finite_readback(&mut kernels, Dtype::F32, &got);
}

/// A `scale` of zero and a negative scale are refused (packet FUSED-KV8-a), and
/// each refusal leaves the adapter usable: the same call with a positive scale succeeds.
#[test]
fn quantized_attention_decode_refuses_nonpositive_scale() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, _kv_heads, width, group) = (4u32, 2u32, 64u32, 64u32);
    let len = 32u32;
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    let (query, keys, values) = negative_case(&mut kernels);
    let keys_view = QuantizedKv {
        codes: &keys.0,
        scales: &keys.1,
        biases: &keys.2,
    };
    let values_view = QuantizedKv {
        codes: &values.0,
        scales: &values.1,
        biases: &values.2,
    };
    for &bad in &[0.0f32, -0.5] {
        let refused = kernels.quantized_attention_decode(
            &query,
            &keys_view,
            &values_view,
            len,
            bad,
            group,
            8,
        );
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("scale")),
            "scale {bad} must be BadRequest: {:?}",
            refused.err()
        );
    }
    // The adapter stays usable: a positive scale still decodes, and the readback proves
    // the kernel compiled and ran - every value is finite.
    let got = kernels
        .quantized_attention_decode(&query, &keys_view, &values_view, len, scale, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_finite_readback(&mut kernels, Dtype::F32, &got);
}

/// Mixed float dtypes follow the documented rule of `shape::quantized_kv_operands` (packet
/// FUSED-KV8-a): the queries and the four scale and bias residents are each F32,
/// or BF16 after `allow_bf16` - INDEPENDENTLY, so an F32 query over BF16 scales and a BF16
/// query over F32 scales are both accepted - while the codes stay U32 and the scale and bias
/// dtypes must agree within each side and across the two sides. The test asserts exactly
/// that rule: the two mixed pairs decode, a keys side of BF16 scales beside a values side of
/// F32 scales is refused, and a valid call right after the refusal proves the adapter stayed
/// usable.
#[test]
fn quantized_attention_decode_follows_the_documented_mixed_dtype_rule() {
    let _serial = GPU_TEST.lock().unwrap_or_else(|poison| poison.into_inner());
    let (heads, kv_heads, width, group) = (4u32, 2u32, 64u32, 64u32);
    let (cap, len) = (64u32, 32u32);
    let scale = 1.0 / (width as f32).sqrt();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);
    let f32_query = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, width], query_value()),
    );
    let bf16_query = kernels.cast(&f32_query, Dtype::Bf16).unwrap();
    let f32_keys = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        key_value(width, cap, false),
    );
    let f32_values = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::F32,
        value_value(width, cap, false),
    );
    let bf16_keys = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::Bf16,
        key_value(width, cap, false),
    );
    let bf16_values = quantized_capacity_of_dtype(
        &mut kernels,
        kv_heads,
        cap,
        width,
        group,
        Dtype::Bf16,
        value_value(width, cap, false),
    );
    fn view<'a>(side: &'a (ExecutionTensor, ExecutionTensor, ExecutionTensor)) -> QuantizedKv<'a> {
        QuantizedKv {
            codes: &side.0,
            scales: &side.1,
            biases: &side.2,
        }
    }
    // The rule accepts each mixed pair: F32 queries over BF16 scales, BF16 queries over F32
    // scales. The output dtype follows the queries.
    let got = kernels
        .quantized_attention_decode(
            &f32_query,
            &view(&bf16_keys),
            &view(&bf16_values),
            len,
            scale,
            group,
            8,
        )
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_eq!(got.dtype(), Dtype::F32);
    assert_finite_readback(&mut kernels, Dtype::F32, &got);
    let got = kernels
        .quantized_attention_decode(
            &bf16_query,
            &view(&f32_keys),
            &view(&f32_values),
            len,
            scale,
            group,
            8,
        )
        .unwrap();
    assert_eq!(got.dtype(), Dtype::Bf16);
    assert_finite_readback(&mut kernels, Dtype::Bf16, &got);

    // The rule refuses a disagreement ACROSS the sides: BF16 key scales beside F32 value
    // scales do not agree.
    let refused = kernels.quantized_attention_decode(
        &f32_query,
        &view(&bf16_keys),
        &view(&f32_values),
        len,
        scale,
        group,
        8,
    );
    assert!(
        matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("must agree")),
        "sides of different scale dtypes must be BadRequest: {:?}",
        refused.err()
    );
    // The adapter stays usable: the matching F32 pair right after the refusal decodes, and
    // the readback proves the kernel compiled and ran - every value is finite.
    let got = kernels
        .quantized_attention_decode(
            &f32_query,
            &view(&f32_keys),
            &view(&f32_values),
            len,
            scale,
            group,
            8,
        )
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);
    assert_finite_readback(&mut kernels, Dtype::F32, &got);
}

/// The decision 0032 fault hooks of the fused decode shim (packet FUSED-KV8-a test d):
/// `call`, `null`, and `mismatch` on the first selected call. Every native failure is
/// terminal for the process, so each case runs in its own child process, as the fault tests
/// of the KV8-OPS operations do.
#[cfg(feature = "native-fault-qualification")]
mod fault_hooks {
    use std::{
        env,
        io::Read,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    use inference_engine::{
        EngError,
        kernels::{MlxKernels, QuantizedKv},
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    use super::{f32_pattern, import};

    /// A deterministic U32 tensor of `shape` from a value function, the codes of the tiny
    /// operands below.
    fn u32_pattern(shape: &[u32], value: impl Fn(usize) -> u32) -> Tensor {
        let count: usize = shape.iter().map(|&dim| dim as usize).product();
        Tensor::try_from_host(
            Dtype::U32,
            shape.to_vec(),
            (0..count)
                .flat_map(|index| value(index).to_le_bytes())
                .collect(),
        )
        .unwrap()
    }

    const CHILD: &str = "INFERENCE_Q8_DECODE_FAULT_CHILD";
    const DEADLINE: Duration = Duration::from_secs(180);

    fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// In the parent, runs `name` in a child that carries `fault`. Returns whether the caller
    /// is the parent and has nothing left to do.
    fn delegated_to_child(name: &str, fault: (&str, &str)) -> bool {
        if env::var_os(CHILD).is_some() {
            return false;
        }
        let path = module_path!().split_once("::").unwrap().1;
        let mut child = Command::new(env::current_exe().unwrap())
            .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
            .env(CHILD, "1")
            .env(fault.0, fault.1)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = drain(child.stdout.take().unwrap());
        let stderr = drain(child.stderr.take().unwrap());
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if started.elapsed() >= DEADLINE {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
        assert!(
            status.is_some_and(|status| status.success()) && captured.contains("1 passed"),
            "child did not pass {name}: {captured}"
        );
        true
    }

    fn expect_backend<T>(result: Result<T, EngError>, label: &str) {
        match result {
            Err(EngError::Backend { operation, .. }) => assert_eq!(
                operation, label,
                "the failure must carry the operation label {label}"
            ),
            Err(other) => panic!("expected a Backend failure of {label}, got {other:?}"),
            Ok(_) => panic!("expected a Backend failure of {label}, got Ok"),
        }
    }

    /// After a native failure every later call of the adapter is refused; `probe` is imported
    /// before the failing call, because imports are refused too.
    fn terminal_check(kernels: &mut MlxKernels, probe: &ExecutionTensor) {
        assert!(
            matches!(kernels.silu(probe), Err(EngError::Backend { .. })),
            "a native failure must make the adapter terminal"
        );
    }

    /// One fused decode call over a tiny pair: KVH 1, QH 2, D 64, cap 4.
    fn decode_call(
        kernels: &mut MlxKernels,
        operands: &[ExecutionTensor; 7],
    ) -> Result<ExecutionTensor, EngError> {
        let keys = QuantizedKv {
            codes: &operands[1],
            scales: &operands[2],
            biases: &operands[3],
        };
        let values = QuantizedKv {
            codes: &operands[4],
            scales: &operands[5],
            biases: &operands[6],
        };
        kernels.quantized_attention_decode(&operands[0], &keys, &values, 4, 0.125, 64, 8)
    }

    /// Builds the tiny operands: the query and the three residents of each side. They are
    /// built from imports only: the fault of these tests strikes the NEXT counted native
    /// operation, and a `quantize_rows` call here would take it before the call under test.
    fn tiny_operands(kernels: &mut MlxKernels) -> [ExecutionTensor; 7] {
        let side = |kernels: &mut MlxKernels, salt: u32| -> [ExecutionTensor; 3] {
            let codes = u32_pattern(&[1, 1, 4, 16], |index| {
                (index as u32).wrapping_mul(0x0101_0101).wrapping_add(salt)
            });
            let scales = f32_pattern(&[1, 1, 4, 1], |index| 0.02 + 0.01 * index as f32);
            let biases = f32_pattern(&[1, 1, 4, 1], |index| -0.5 + 0.1 * index as f32);
            [
                import(kernels, &codes),
                import(kernels, &scales),
                import(kernels, &biases),
            ]
        };
        let query = import(
            kernels,
            &f32_pattern(&[1, 2, 1, 64], |index| ((index % 7) as f32 - 3.0) / 4.0),
        );
        let [k_codes, k_scales, k_biases] = side(kernels, 0);
        let [v_codes, v_scales, v_biases] = side(kernels, 0x0f0f_0f0f);
        [
            query, k_codes, k_scales, k_biases, v_codes, v_scales, v_biases,
        ]
    }

    #[test]
    fn quantized_attention_decode_call_failure_is_terminal() {
        if delegated_to_child(
            "quantized_attention_decode_call_failure_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            decode_call(&mut kernels, &operands),
            "quantized attention decode",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantized_attention_decode_null_output_is_terminal() {
        if delegated_to_child(
            "quantized_attention_decode_null_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            decode_call(&mut kernels, &operands),
            "quantized attention decode",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantized_attention_decode_mismatch_is_terminal() {
        if delegated_to_child(
            "quantized_attention_decode_mismatch_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        // The widened output no longer matches the checked layout, so the metadata check
        // fails.
        expect_backend(
            decode_call(&mut kernels, &operands),
            "quantized attention decode",
        );
        terminal_check(&mut kernels, &probe);
    }
}
