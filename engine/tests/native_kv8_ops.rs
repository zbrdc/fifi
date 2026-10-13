//! Native tests of the KV8-OPS operations: `quantize_rows`, the U32 `kv_append`, and
//! `quantized_attention_prefix` over the 8-bit KV cache (packet KV8-OPS).

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

/// A deterministic U32 tensor of `shape` from a value function.
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

/// Reads U32 values out of raw readback bytes.
fn u32_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

fn u32_values(tensor: &Tensor) -> Vec<u32> {
    u32_from_bytes(tensor.host_bytes().unwrap())
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

/// Compares two f32 slices within 1e-5 relative to max(1, |expected|).
fn assert_within(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: {actual:?}");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "{context} element {index}: actual {a}, expected {b}"
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

/// Dequantizes one whole quantized block through the existing `dequantize` operation: the three
/// residents are read back to host tensors and imported as one `PackedWeight` of logical shape
/// `logical` (whose last axis is the F32 width), and `dequantize` returns the F32
/// reconstruction. This is the independent oracle the packet names: it runs MLX's own
/// dequantize on MLX's own quantize outputs.
fn dequantized_block(
    kernels: &mut MlxKernels,
    codes: &ExecutionTensor,
    scales: &ExecutionTensor,
    biases: &ExecutionTensor,
    logical: Vec<u32>,
    group: u32,
) -> Tensor {
    let codes_host = read(kernels, codes);
    let scales_host = read(kernels, scales);
    let biases_host = read(kernels, biases);
    let packed = PackedWeight::try_new(
        codes_host,
        scales_host,
        Some(biases_host),
        logical,
        affine(group),
    )
    .unwrap();
    let resident = kernels.import_packed(&packed).unwrap();
    let dequantized = kernels.dequantize(&resident).unwrap();
    read(kernels, &dequantized)
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
/// `dequantize` operation, as `dequantized_block` does: an F32 `[1, KVH, len, D]` resident that
/// `grouped_attention` consumes as the oracle of the quantized prefix.
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

/// `quantize_rows` (packet KV8-OPS test a) against an independent f64 host check. MLX's affine
/// definition is not the plain `scale = (max - min) / 255`, `bias = min`: it moves the larger
/// edge of each group of 64 lanes onto a code, so the scale can come out negative and its
/// magnitude differs from the plain figure by at most 0.5 / 127 (first Mac run, 2026-10-13: scale
/// -2.4710e-4 against the plain 2.4706e-4). The test pins what any such definition must satisfy:
/// the magnitude of the scale within one percent of `(max - min) / 255`; the packed code of each
/// lane (four 8-bit codes per U32 word, the low lane first) equal to the rounded quotient
/// `(x - bias) / scale` of the scale and bias that the call returned; the existing `dequantize`
/// of the outputs equal to the f64 reconstruction `code * scale + bias` of the SAME codes,
/// scales and biases; and the quantize-dequantize error of each element at most `|scale| / 2`
/// plus float rounding (the F32 rounding of the scale alone moves the reconstruction by at most
/// `255 * |scale| * 2^-24`, about `|scale| * 1.6e-5`). The rows carry a spread of magnitudes, one
/// CONSTANT group (every code reconstructs to the bias), one group with a single outlier, and a
/// signed row.
#[test]
fn quantize_rows_matches_the_f64_affine_oracle() {
    let _serial = GPU_TEST.lock().unwrap();
    let group = 64u32;
    let rows: Vec<Vec<f32>> = vec![
        // A magnitude spread: 1e-3 scale, then ~1e2.
        (0..128)
            .map(|lane| {
                if lane < 64 {
                    0.001 * (lane as f32 + 1.0)
                } else {
                    100.0 + (lane - 64) as f32
                }
            })
            .collect(),
        // A constant group (scale 0) and a group with one outlier.
        (0..128)
            .map(|lane| {
                if lane < 64 {
                    0.5
                } else if lane == 64 {
                    500.0
                } else {
                    (lane - 65) as f32 / 8.0
                }
            })
            .collect(),
        // A signed row.
        (0..128)
            .map(|lane| ((lane % 17) as f32 - 8.0) * ((lane % 5) as f32 + 1.0))
            .collect(),
    ];
    let host: Vec<f32> = rows.concat();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = import(&mut kernels, &f32_input(vec![3, 128], &host));

    let (codes, scales, biases) = kernels.quantize_rows(&x, group, 8).unwrap();
    assert_eq!(
        codes.shape(),
        [3, 32],
        "128 lanes of 8 bits pack into 32 words"
    );
    assert_eq!(codes.dtype(), Dtype::U32);
    assert_eq!(codes.byte_len(), 3 * 32 * 4);
    assert_eq!(scales.shape(), [3, 2]);
    assert_eq!(scales.dtype(), Dtype::F32, "F32 rows keep F32 scales");
    assert_eq!(biases.shape(), [3, 2]);
    assert_eq!(scales.byte_len(), 3 * 2 * 4);

    let codes_host = u32_values(&read(&mut kernels, &codes));
    let scales_host = values_of(&read(&mut kernels, &scales));
    let biases_host = values_of(&read(&mut kernels, &biases));
    let dequantized = values_of(&dequantized_block(
        &mut kernels,
        &codes,
        &scales,
        &biases,
        vec![3, 128],
        group,
    ));

    for r in 0..3usize {
        for g in 0..2usize {
            let lanes: Vec<f64> = host[r * 128 + g * 64..r * 128 + g * 64 + 64]
                .iter()
                .map(|value| f64::from(*value))
                .collect();
            let min = lanes.iter().copied().fold(f64::INFINITY, f64::min);
            let max = lanes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = max - min;
            let scale = f64::from(scales_host[r * 2 + g]);
            let bias = f64::from(biases_host[r * 2 + g]);
            // MLX's affine definition is not the plain (max - min) / 255: it moves the larger
            // edge of the group onto a code (the scale may even come out negative), which
            // changes the magnitude of the scale by at most 0.5 / 127 (the larger edge is at
            // least half of the span, so its code is at least 127). The test therefore pins the
            // MAGNITUDE to one percent and the reconstruction to half a step, not the sign.
            let step = scale.abs();
            if span > 0.0 {
                assert!(
                    (step - span / 255.0).abs() <= span / 255.0 * 0.01,
                    "row {r} group {g}: |scale| {step} against (max - min) / 255 = {}",
                    span / 255.0
                );
            }
            for (lane, &oracle_value) in lanes.iter().enumerate() {
                let flat = r * 128 + g * 64 + lane;
                // Four 8-bit codes per word, the low lane first.
                let word = codes_host[flat / 4];
                let code = f64::from((word >> (8 * (flat % 4))) & 0xff);
                // The packed code is the rounded quotient of the value by the scale and bias
                // that the call returned; a wrong lane order or a wrong word breaks it.
                if step > 1e-6 {
                    let recomputed = ((oracle_value - bias) / scale).round().clamp(0.0, 255.0);
                    assert!(
                        (code - recomputed).abs() <= 1.0,
                        "row {r} group {g} lane {lane}: code {code} against the rounded \
                         quotient {recomputed} (scale {scale}, bias {bias})"
                    );
                }
                let reconstruction = code * scale + bias;
                let value = f64::from(host[flat]);
                assert!(
                    (f64::from(dequantized[flat]) - reconstruction).abs()
                        <= 1e-5 * reconstruction.abs().max(1.0),
                    "row {r} group {g} lane {lane}: dequantize {} against the reconstruction \
                     {reconstruction}",
                    dequantized[flat]
                );
                assert!(
                    (value - reconstruction).abs() <= step * 0.5 + step * 2e-5 + 1e-9,
                    "row {r} group {g} lane {lane}: value {value} against the reconstruction \
                     {reconstruction} at |scale| {step}"
                );
            }
        }
    }
}

/// One quantized capacity side from an F32 value function: the rows `[1, kv_heads, rows,
/// width]` are quantized by `quantize_rows`, and the three residents keep the capacity layout.
fn quantized_capacity(
    kernels: &mut MlxKernels,
    kv_heads: u32,
    rows: u32,
    width: u32,
    group: u32,
    value: impl Fn(usize) -> f32,
) -> (ExecutionTensor, ExecutionTensor, ExecutionTensor) {
    let source = import(kernels, &f32_pattern(&[1, kv_heads, rows, width], value));
    let (codes, scales, biases) = kernels.quantize_rows(&source, group, 8).unwrap();
    assert_eq!(codes.shape(), [1, kv_heads, rows, width / 4]);
    assert_eq!(scales.shape(), [1, kv_heads, rows, width / group]);
    (codes, scales, biases)
}

/// `quantized_attention_prefix` against `grouped_attention` over the DEQUANTIZED keys and
/// values (packet KV8-OPS test b): the dequantized prefix is the independent oracle - the same
/// math on the same numbers - so the two agree within float accumulation, and the tolerance
/// of `assert_within` covers that error, derived: every F32 product and partial sum rounds to
/// 2^-24 relative, and the mixed-sign patterns (period 7 over period 5, whose 35-lane period
/// sums to exactly zero) keep the partial sums within about 2, so the D <= 256 lane score dot
/// of each path wanders at most sqrt(256) · 2^-24 · 2 ≈ 2e-6 and the two paths (the quantized
/// matmul and the fused tiling) stay ≈ 4e-6 apart per score, ≈ 5e-7 per logit after the scale
/// multiply (scale <= 1/8). The softmax damps a logit error eps into a probability error of at
/// most 2 · eps · p, so the |v| <= 1 value mix moves <= 1e-6, and the len <= 16 lane value mix
/// and the two output roundings add about sqrt(16) · 2^-24 ≈ 2.4e-7 each - under 2e-6 in
/// total, five times below the 1e-5 relative-to-max(1, |expected|) tolerance. L = 1 non-causal
/// and L = 7 causal, len 1, 5, CAP - 1, CAP, GQA 16 over 2 heads and a small 4 over 2, head
/// dim 64 and one case at 256, group 64. A causal chunk needs len >= L - the portable check
/// rejects a shorter one, whose earliest rows would see no key - so the sweep keeps
/// len >= L as before.
#[test]
fn quantized_attention_prefix_matches_the_dequantized_prefix() {
    let _serial = GPU_TEST.lock().unwrap();
    let cap = 16u32;
    let group = 64u32;
    for &(heads, kv_heads, width, causal, query_rows, lens) in &[
        (16u32, 2u32, 64u32, false, 1u32, &[1u32, 5, 15, 16][..]),
        (16u32, 2u32, 64u32, true, 7u32, &[7u32, 8, 15, 16][..]),
        (4u32, 2u32, 64u32, true, 7u32, &[7u32, 15][..]),
        (4u32, 2u32, 256u32, true, 7u32, &[7u32, 16][..]),
    ] {
        let mut kernels = MlxKernels::try_new().unwrap();
        let q = import(
            &mut kernels,
            &f32_pattern(&[1, heads, query_rows, width], |index| {
                (((index % 7) as f32) - 3.0) / 4.0
            }),
        );
        let (k_codes, k_scales, k_biases) =
            quantized_capacity(&mut kernels, kv_heads, cap, width, group, |index| {
                (((index % 5) as f32) - 2.0) / 4.0
            });
        let (v_codes, v_scales, v_biases) =
            quantized_capacity(&mut kernels, kv_heads, cap, width, group, |index| {
                (((index * 3 % 9) as f32) - 4.0) / 4.0
            });
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
        let scale = 1.0 / (width as f32).sqrt();
        for &len in lens {
            let oracle_k = dequantized_prefix(&mut kernels, &keys, len, group);
            let oracle_v = dequantized_prefix(&mut kernels, &values, len, group);
            let expected = kernels
                .grouped_attention(&q, &oracle_k, &oracle_v, scale, causal)
                .unwrap();
            let got = kernels
                .quantized_attention_prefix(&q, &keys, &values, len, scale, causal, group, 8)
                .unwrap();
            assert_eq!(
                got.shape(),
                [1, heads, query_rows, width],
                "heads {heads} over {kv_heads}, width {width}, causal {causal}, len {len}"
            );
            assert_eq!(got.dtype(), Dtype::F32, "F32 queries give an F32 result");
            let expected_values = values_of(&read(&mut kernels, &expected));
            let got_values = values_of(&read(&mut kernels, &got));
            assert_within(
                &got_values,
                &expected_values,
                &format!(
                    "heads {heads} over {kv_heads}, width {width}, causal {causal}, len {len}"
                ),
            );
        }
    }
}

/// One quantized capacity side whose scales and biases carry `dtype`: the F32 source of
/// `value` is imported and cast first, so `quantize_rows` returns scales and biases of `dtype`
/// beside the U32 codes - F32 stays itself, BF16 gives the BF16 prefix of the alignment case.
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

/// Reconstructs the first `len` rows of one one-KV-head quantized side on the host in f64:
/// `code * scale + bias` per group of `group` lanes, the codes unpacked from the low end,
/// four 8-bit codes per U32 word - the packing `quantize_rows_matches_the_f64_affine_oracle`
/// pins. `scales` and `biases` are already decoded to f32 (an F32 or BF16 readback).
fn dequantized_rows(
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    len: u32,
    width: u32,
    group: u32,
) -> Vec<f64> {
    let words_per_row = (width / 4) as usize;
    let groups_per_row = (width / group) as usize;
    let mut rows = Vec::with_capacity(len as usize * width as usize);
    for row in 0..len as usize {
        for lane in 0..width as usize {
            let word = codes[row * words_per_row + lane / 4];
            let code = f64::from((word >> (8 * (lane % 4))) & 0xFF);
            let scale = f64::from(scales[row * groups_per_row + lane / group as usize]);
            let bias = f64::from(biases[row * groups_per_row + lane / group as usize]);
            rows.push(code * scale + bias);
        }
    }
    rows
}

/// One causal-alignment case against the f64 host softmax over the DEQUANTIZED keys and
/// values: QH 2 over KVH 1 at head dim `width`, `rows` query rows, `len` of `cap` keys, F32 or
/// BF16 (`bf16` gives BF16 queries and BF16 scales and biases after `allow_bf16`, so the
/// scores, the `where` selection, and the output all stay BF16). The oracle reconstructs the
/// first `len` rows of each side on the host and computes the masked softmax per query row:
/// row t sees keys 0..=len - rows + t, so a mask aligned elsewhere, inverted, or dropped moves
/// whole probabilities, far beyond the comparison bound. The F32 bound is the 1e-5 of
/// `assert_within`; the BF16 prefix rounds at 2^-8 relative per value (about 4e-3 on the O(1)
/// scores, probabilities, and values of these patterns), and the score-softmax-mix
/// accumulation stays an order below the 5e-2 bound.
fn check_causal_alignment(
    kernels: &mut MlxKernels,
    rows: u32,
    len: u32,
    cap: u32,
    width: u32,
    group: u32,
    bf16: bool,
) {
    if bf16 {
        kernels.allow_bf16(true);
    }
    let scale = 1.0 / (width as f32).sqrt();
    let dtype = if bf16 { Dtype::Bf16 } else { Dtype::F32 };
    let decode = |tensor: &Tensor| -> Vec<f32> {
        if bf16 {
            bf16_values(tensor)
        } else {
            values_of(tensor)
        }
    };
    let mut query = import(
        kernels,
        &f32_pattern(&[1, 2, rows, width], |index| {
            (((index % 7) as f32) - 3.0) / 4.0
        }),
    );
    if bf16 {
        query = kernels.cast(&query, Dtype::Bf16).unwrap();
    }
    let (k_codes, k_scales, k_biases) =
        quantized_capacity_of_dtype(kernels, 1, cap, width, group, dtype, |index| {
            (((index % 5) as f32) - 2.0) / 4.0
        });
    let (v_codes, v_scales, v_biases) =
        quantized_capacity_of_dtype(kernels, 1, cap, width, group, dtype, |index| {
            (((index * 3 % 9) as f32) - 4.0) / 4.0
        });
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

    // The f64 oracle over the SAME numbers the kernel computes with: the readback codes,
    // scales, and biases of the first `len` rows, reconstructed on the host.
    let k = dequantized_rows(
        &u32_values(&first_rows(&read(kernels, &k_codes), len)),
        &decode(&first_rows(&read(kernels, &k_scales), len)),
        &decode(&first_rows(&read(kernels, &k_biases), len)),
        len,
        width,
        group,
    );
    let v = dequantized_rows(
        &u32_values(&first_rows(&read(kernels, &v_codes), len)),
        &decode(&first_rows(&read(kernels, &v_scales), len)),
        &decode(&first_rows(&read(kernels, &v_biases), len)),
        len,
        width,
        group,
    );
    let query_values = decode(&read(kernels, &query));

    let got = kernels
        .quantized_attention_prefix(&query, &keys, &values, len, scale, true, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, 2, rows, width]);
    assert_eq!(
        got.dtype(),
        dtype,
        "the where selection keeps the scores' dtype"
    );
    let got_values = decode(&read(kernels, &got));
    // Query row t sees keys 0..=(len - rows + t): row 0 the oldest len - rows + 1 keys, and
    // the last row every key of the prefix.
    let tolerance = if bf16 { 5e-2 } else { 1e-5 };
    for head in 0..2usize {
        for t in 0..rows as usize {
            let visible = (len - rows + t as u32) as usize + 1;
            let logits: Vec<f64> = (0..visible)
                .map(|key| {
                    let dot: f64 = (0..width as usize)
                        .map(|lane| {
                            f64::from(
                                query_values[(head * rows as usize + t) * width as usize + lane],
                            ) * k[key * width as usize + lane]
                        })
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logits.iter().map(|logit| (logit - top).exp()).sum();
            for lane in 0..width as usize {
                let weighted: f64 = logits
                    .iter()
                    .enumerate()
                    .map(|(key, logit)| {
                        ((logit - top).exp() / total) * v[key * width as usize + lane]
                    })
                    .sum();
                let expected = weighted as f32;
                let actual = got_values[(head * rows as usize + t) * width as usize + lane];
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "head {head} row {t} lane {lane}: actual {actual}, expected {expected}"
                );
            }
        }
    }
}

/// The causal alignment of the quantized prefix against an f64 host softmax over the
/// DEQUANTIZED keys and values (packet KV8-OPS test c, the form of the
/// `grouped_attention_prefix` alignment test): the mask aligns to the LAST key of the prefix,
/// so query row `t` of `L` sees keys `0..len - L + t`. A mask aligned to the FIRST key would
/// let query 0 see only key 0. Three cases at QH 2 over KVH 1, head dim 64, group 64: F32
/// with L 2, len 3 of a capacity 4 (one key more than the rows); F32 with L 3, len 6 of a
/// capacity 8 (the gap len - L larger than one, the alignment exercised with rows > 2); and a
/// BF16 prefix - BF16 queries and BF16 scales and biases - with L 3, len 5 of a capacity 8,
/// whose `where`-masked scores keep BF16 and whose result stays BF16.
#[test]
fn quantized_attention_prefix_aligns_the_causal_mask_to_the_last_key() {
    let _serial = GPU_TEST.lock().unwrap();
    let group = 64u32;
    let mut kernels = MlxKernels::try_new().unwrap();
    check_causal_alignment(&mut kernels, 2, 3, 4, 64, group, false);
    check_causal_alignment(&mut kernels, 3, 6, 8, 64, group, false);
    check_causal_alignment(&mut kernels, 3, 5, 8, 64, group, true);
}

/// Appends row `t` of `source` into `target` at position `t` (the pipeline loop of test d).
fn append_row(
    kernels: &mut MlxKernels,
    target: ExecutionTensor,
    source: &ExecutionTensor,
    t: u32,
    last: u32,
) -> ExecutionTensor {
    let row = kernels
        .slice(source, &[0, 0, t, 0], &[1, source.shape()[1], t + 1, last])
        .unwrap();
    kernels.kv_append(target, &row, &[0, 0, t, 0]).unwrap()
}

/// The pipeline round trip (packet KV8-OPS test d): `quantize_rows` of 20 rows, `kv_append` of
/// the three outputs row by row into three capacity buffers of 32 (the codes through the new
/// U32 rule), and attention over the first 20 rows equals attention over the DEQUANTIZED 20
/// rows of the original quantization. KVH 2, QH 16, head dim 256, group 64, decode queries.
#[test]
fn quantized_attention_prefix_round_trips_quantize_append_and_attention() {
    let _serial = GPU_TEST.lock().unwrap();
    let (kv_heads, heads, width, group, rows, cap) = (2u32, 16u32, 256u32, 64u32, 20u32, 32u32);
    let mut kernels = MlxKernels::try_new().unwrap();
    let (k_codes, k_scales, k_biases) =
        quantized_capacity(&mut kernels, kv_heads, rows, width, group, |index| {
            (((index % 23) as f32) - 11.0) / 8.0
        });
    let (v_codes, v_scales, v_biases) =
        quantized_capacity(&mut kernels, kv_heads, rows, width, group, |index| {
            (((index * 3 % 19) as f32) - 9.0) / 8.0
        });
    let q = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, width], |index| {
            (((index % 15) as f32) - 7.0) / 8.0
        }),
    );

    // The capacity buffers start at zero; row `t` of each output lands at position `t`.
    let zeros_u32 = u32_pattern(&[1, kv_heads, cap, width / 4], |_| 0);
    let zeros_f32 = f32_pattern(&[1, kv_heads, cap, width / group], |_| 0.0);
    let mut key_codes = import(&mut kernels, &zeros_u32);
    let mut key_scales = import(&mut kernels, &zeros_f32);
    let mut key_biases = import(&mut kernels, &zeros_f32);
    let mut value_codes = import(&mut kernels, &zeros_u32);
    let mut value_scales = import(&mut kernels, &zeros_f32);
    let mut value_biases = import(&mut kernels, &zeros_f32);
    let (words, groups) = (width / 4, width / group);
    for t in 0..rows {
        key_codes = append_row(&mut kernels, key_codes, &k_codes, t, words);
        key_scales = append_row(&mut kernels, key_scales, &k_scales, t, groups);
        key_biases = append_row(&mut kernels, key_biases, &k_biases, t, groups);
        value_codes = append_row(&mut kernels, value_codes, &v_codes, t, words);
        value_scales = append_row(&mut kernels, value_scales, &v_scales, t, groups);
        value_biases = append_row(&mut kernels, value_biases, &v_biases, t, groups);
    }
    assert_eq!(key_codes.dtype(), Dtype::U32, "the codes append as U32");

    let keys = QuantizedKv {
        codes: &key_codes,
        scales: &key_scales,
        biases: &key_biases,
    };
    let values = QuantizedKv {
        codes: &value_codes,
        scales: &value_scales,
        biases: &value_biases,
    };
    let scale = 1.0 / (width as f32).sqrt();
    let got = kernels
        .quantized_attention_prefix(&q, &keys, &values, rows, scale, false, group, 8)
        .unwrap();
    assert_eq!(got.shape(), [1, heads, 1, width]);

    // The oracle: the same attention over the dequantized 20 rows of the original outputs.
    let original_keys = QuantizedKv {
        codes: &k_codes,
        scales: &k_scales,
        biases: &k_biases,
    };
    let original_values = QuantizedKv {
        codes: &v_codes,
        scales: &v_scales,
        biases: &v_biases,
    };
    let oracle_k = dequantized_prefix(&mut kernels, &original_keys, rows, group);
    let oracle_v = dequantized_prefix(&mut kernels, &original_values, rows, group);
    let expected = kernels
        .grouped_attention(&q, &oracle_k, &oracle_v, scale, false)
        .unwrap();
    assert_within(
        &values_of(&read(&mut kernels, &got)),
        &values_of(&read(&mut kernels, &expected)),
        "the appended quantized cache against the dequantized rows",
    );
}

/// The mean wall time of ten readback-complete rounds of `build`, after one warm-up round.
fn timed_rounds(
    kernels: &mut MlxKernels,
    label: &str,
    mut build: impl FnMut(&mut MlxKernels) -> ExecutionTensor,
) -> f64 {
    let warm = build(kernels);
    kernels.readback(&warm).unwrap();
    drop(warm);
    let began = Instant::now();
    for _ in 0..10 {
        let output = build(kernels);
        kernels.readback(&output).unwrap();
    }
    let milliseconds = began.elapsed().as_secs_f64() / 10.0 * 1e3;
    println!("quantized prefix view cost, {label}: {milliseconds:.3} ms per call");
    milliseconds
}

/// The VIEW COST measurement (packet KV8-OPS test f): CAP 32,768, len 16,384, two KV heads,
/// head dim 256, 8-bit, group 64. What the numbers mean: the six prefix views are strided
/// across the head axis (a head's stride is CAP * last, not len * last), so a
/// `quantized_matmul` that copies a non-row-contiguous operand would copy about 8 MiB of K
/// codes plus 0.5 MiB each of K scales and biases, and the same for V - about 18 MiB - on
/// EVERY call, which at 262K positions would defeat the whole 8-bit cache. The strided call
/// itself allocates only the output (16 KiB), the scores, the scaled scores, and the
/// probabilities (three `[1, 2, 8, 16384]` F32 buffers, 1 MiB each) plus softmax scratch, so
/// its peak growth stays far below 16 MiB; the asserted bound is loose - it sits above that
/// honest scratch and below the 18 MiB that one view copy would add. The contiguous control
/// copies the six prefixes first (`slice` makes them row-contiguous) and must show the copies
/// in the same metric, which proves the metric sees a copy; the len = CAP call measures the
/// full-range view, whose head stride equals the row span, so it is row-contiguous everywhere.
/// The wall times of all three paths are printed for the coordinator; only the peak bounds are
/// asserted. If the strided call is not much slower than the contiguous copy-free time and its
/// peak growth stays below the bound, MLX did not copy the views.
///
/// History (Mac, 2026-10-13): the first run read a peak growth of 38.8 MB and concluded that MLX
/// copies the strided views. That was a measurement error: the six buffers were still lazy, so
/// the first call also materialized the quantization of the cache. A probe in the MLX Python
/// package on the same shapes (a quantized matmul over the full capacity, over the prefix slices
/// of all heads, and over per-head slices) grew the peak by 1 to 2 MB in every form and ran
/// faster on the prefix than on the full buffer: `quantized_matmul` reads a strided prefix view
/// in place. The buffers are now evaluated before the first measurement.
#[test]
fn quantized_attention_prefix_view_cost_at_half_capacity() {
    let _serial = GPU_TEST.lock().unwrap();
    let (cap, len, kv_heads, heads, width, group) =
        (32768u32, 16384u32, 2u32, 16u32, 256u32, 64u32);
    let mut kernels = MlxKernels::try_new().unwrap();
    let (k_codes, k_scales, k_biases) =
        quantized_capacity(&mut kernels, kv_heads, cap, width, group, |index| {
            (((index % 23) as f32) - 11.0) / 8.0
        });
    let (v_codes, v_scales, v_biases) =
        quantized_capacity(&mut kernels, kv_heads, cap, width, group, |index| {
            (((index * 3 % 19) as f32) - 9.0) / 8.0
        });
    let q = import(
        &mut kernels,
        &f32_pattern(&[1, heads, 1, width], |index| {
            (((index % 15) as f32) - 7.0) / 8.0
        }),
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
    let scale = 1.0 / (width as f32).sqrt();

    // The six buffers and the queries are LAZY until the first evaluation: without this
    // evaluation the first measured call would also materialize the quantization of the whole
    // cache (the F32 source and the three outputs of each side) and its peak would read as a
    // copy of the views. The first Mac run of this test made exactly that mistake (38.8 MB).
    kernels
        .evaluate(&[
            &q, &k_codes, &k_scales, &k_biases, &v_codes, &v_scales, &v_biases,
        ])
        .unwrap();

    // The peak growth of one strided prefix attention, read back complete.
    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let probe = kernels
        .quantized_attention_prefix(&q, &keys, &values, len, scale, false, group, 8)
        .unwrap();
    kernels.readback(&probe).unwrap();
    drop(probe);
    let strided_peak = kernels.memory_stats().unwrap().peak.saturating_sub(base);

    // The control: the same attention over contiguous copies of the six prefixes.
    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let contiguous = {
        let mut copy = |tensor: &ExecutionTensor| {
            kernels
                .slice(
                    tensor,
                    &[0, 0, 0, 0],
                    &[1, kv_heads, len, tensor.shape()[3]],
                )
                .unwrap()
        };
        let copied_k_codes = copy(&k_codes);
        let copied_k_scales = copy(&k_scales);
        let copied_k_biases = copy(&k_biases);
        let copied_v_codes = copy(&v_codes);
        let copied_v_scales = copy(&v_scales);
        let copied_v_biases = copy(&v_biases);
        let copied_keys = QuantizedKv {
            codes: &copied_k_codes,
            scales: &copied_k_scales,
            biases: &copied_k_biases,
        };
        let copied_values = QuantizedKv {
            codes: &copied_v_codes,
            scales: &copied_v_scales,
            biases: &copied_v_biases,
        };
        kernels
            .quantized_attention_prefix(
                &q,
                &copied_keys,
                &copied_values,
                len,
                scale,
                false,
                group,
                8,
            )
            .unwrap()
    };
    kernels.readback(&contiguous).unwrap();
    drop(contiguous);
    let contiguous_peak = kernels.memory_stats().unwrap().peak.saturating_sub(base);

    // The full-capacity strided call: the view spans every row, so nothing is strided.
    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let full = kernels
        .quantized_attention_prefix(&q, &keys, &values, cap, scale, false, group, 8)
        .unwrap();
    kernels.readback(&full).unwrap();
    drop(full);
    let full_peak = kernels.memory_stats().unwrap().peak.saturating_sub(base);

    timed_rounds(&mut kernels, "strided views at len 16384", |kernels| {
        kernels
            .quantized_attention_prefix(&q, &keys, &values, len, scale, false, group, 8)
            .unwrap()
    });
    timed_rounds(
        &mut kernels,
        "contiguous prefixes at len 16384",
        |kernels| {
            let mut copy = |tensor: &ExecutionTensor| {
                kernels
                    .slice(
                        tensor,
                        &[0, 0, 0, 0],
                        &[1, kv_heads, len, tensor.shape()[3]],
                    )
                    .unwrap()
            };
            let copied_k_codes = copy(&k_codes);
            let copied_k_scales = copy(&k_scales);
            let copied_k_biases = copy(&k_biases);
            let copied_v_codes = copy(&v_codes);
            let copied_v_scales = copy(&v_scales);
            let copied_v_biases = copy(&v_biases);
            let copied_keys = QuantizedKv {
                codes: &copied_k_codes,
                scales: &copied_k_scales,
                biases: &copied_k_biases,
            };
            let copied_values = QuantizedKv {
                codes: &copied_v_codes,
                scales: &copied_v_scales,
                biases: &copied_v_biases,
            };
            kernels
                .quantized_attention_prefix(
                    &q,
                    &copied_keys,
                    &copied_values,
                    len,
                    scale,
                    false,
                    group,
                    8,
                )
                .unwrap()
        },
    );
    timed_rounds(
        &mut kernels,
        "full-capacity views at len 32768",
        |kernels| {
            kernels
                .quantized_attention_prefix(&q, &keys, &values, cap, scale, false, group, 8)
                .unwrap()
        },
    );

    // One copy of the six prefixes at len: (64 code words + 8 scale and bias values) per head
    // and row, 4 bytes each, 2 KV heads, 16384 rows, 2 sides = 18,874,368 bytes.
    let prefix_copy: u64 = (len as u64)
        * u64::from(kv_heads)
        * (u64::from(width / 4) + 2 * u64::from(width / group))
        * 4
        * 2;
    println!(
        "peak growth: strided {strided_peak} bytes, contiguous control {contiguous_peak} bytes, \
         full-capacity {full_peak} bytes; one copy of the six prefixes is {prefix_copy} bytes"
    );

    assert!(
        strided_peak < 16 * 1024 * 1024,
        "MLX copied the prefix views: the strided call grew the peak by {strided_peak} bytes, \
         above the 16 MiB bound (one copy of the six prefixes is about {prefix_copy} bytes)"
    );
    assert!(
        contiguous_peak >= 16 * 1024 * 1024,
        "the contiguous control did not copy the prefixes: peak growth {contiguous_peak} bytes"
    );
}

/// `kv_append` on U32 codes is bit-equal to `slice_update` on the same inputs (packet
/// KV8-OPS): `slice_update` takes float operands, so the control imports the SAME bytes as F32
/// (U32 and F32 share the 4-byte width and both operations only move bytes), and the
/// readback bytes of the two appends must be equal. The landed row and an untouched row are
/// checked from the U32 values.
#[test]
fn kv_append_u32_matches_slice_update_bit_for_bit() {
    let _serial = GPU_TEST.lock().unwrap();
    let target = u32_pattern(&[1, 2, 8, 4], |index| {
        0x0102_0304u32.wrapping_add(index as u32 * 0x1010)
    });
    let update = u32_pattern(&[1, 2, 1, 4], |index| {
        0xfedc_ba98u32.wrapping_sub(index as u32)
    });
    let start = [0u32, 0, 3, 0];
    let mut kernels = MlxKernels::try_new().unwrap();

    // The control: the same bytes as F32 through the retained `slice_update`.
    let target_bytes = target.host_bytes().unwrap().to_vec();
    let update_bytes = update.host_bytes().unwrap().to_vec();
    let as_f32 = |bytes: &[u8], shape: Vec<u32>| {
        Tensor::try_from_host(Dtype::F32, shape, bytes.to_vec()).unwrap()
    };
    let kept = import(&mut kernels, &as_f32(&target_bytes, vec![1, 2, 8, 4]));
    let row = import(&mut kernels, &as_f32(&update_bytes, vec![1, 2, 1, 4]));
    let expected = kernels.slice_update(&kept, &row, &start).unwrap();
    let expected_bytes = read(&mut kernels, &expected).host_bytes().unwrap().to_vec();

    // The consuming path on the U32 tensors.
    let target = import(&mut kernels, &target);
    let update = import(&mut kernels, &update);
    let got = kernels.kv_append(target, &update, &start).unwrap();
    assert_eq!(
        got.shape(),
        [1, 2, 8, 4],
        "the result keeps the target shape"
    );
    assert_eq!(got.dtype(), Dtype::U32);
    assert_eq!(got.byte_len(), 256, "[1, 2, 8, 4] U32 is 64 words");
    let got_bytes = read(&mut kernels, &got).host_bytes().unwrap().to_vec();
    assert_eq!(
        got_bytes, expected_bytes,
        "a U32 kv_append must match slice_update on the same bytes"
    );

    // The landed row and one untouched row: element (g, t, d) sits at (g * 8 + t) * 4 + d.
    let got_values = u32_from_bytes(&got_bytes);
    let update_values = u32_values(&read(&mut kernels, &update));
    for g in 0..2usize {
        for d in 0..4usize {
            assert_eq!(
                got_values[(g * 8 + 3) * 4 + d],
                update_values[g * 4 + d],
                "row 3 of head {g} lane {d} did not take the update"
            );
            assert_eq!(
                got_values[(g * 8 + 2) * 4 + d],
                0x0102_0304u32.wrapping_add(((g * 8 + 2) * 4 + d) as u32 * 0x1010),
                "row 2 of head {g} lane {d} changed"
            );
        }
    }
}

/// A mixed U32/F32 pair is `BadRequest` before any native call, in both directions, and the
/// adapter stays usable (packet KV8-OPS). A refusal consumes no buffer work, and each refusal
/// gets a fresh target because `kv_append` takes its target by value.
#[test]
fn kv_append_rejects_mixed_u32_and_f32_dtypes() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let f32_row = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 1, 4], |index| 100.0 + index as f32),
    );
    let u32_row = import(
        &mut kernels,
        &u32_pattern(&[1, 2, 1, 4], |index| 100 + index as u32),
    );

    let fresh_u32 = import(
        &mut kernels,
        &u32_pattern(&[1, 2, 8, 4], |index| index as u32),
    );
    let refused = kernels.kv_append(fresh_u32, &f32_row, &[0, 0, 3, 0]);
    assert!(
        matches!(refused, Err(EngError::BadRequest(ref message)) if message.contains("dtype")),
        "a U32 target with an F32 row must be BadRequest: {:?}",
        refused.err()
    );

    let fresh_f32 = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 8, 4], |index| index as f32),
    );
    let refused = kernels.kv_append(fresh_f32, &u32_row, &[0, 0, 3, 0]);
    assert!(
        matches!(refused, Err(EngError::BadRequest(ref message)) if message.contains("dtype")),
        "an F32 target with a U32 row must be BadRequest: {:?}",
        refused.err()
    );

    // The adapter stays usable: a matching U32 pair still appends.
    let target = import(
        &mut kernels,
        &u32_pattern(&[1, 2, 8, 4], |index| index as u32),
    );
    let got = kernels.kv_append(target, &u32_row, &[0, 0, 3, 0]).unwrap();
    let values = u32_values(&read(&mut kernels, &got));
    assert_eq!(values[3 * 4], 100, "a later matching append still works");
}

/// `kv_append` donates a U32 buffer (packet KV8-OPS): 200 appends of one U32 row into one
/// `[1, 2, 16384, 256]` U32 buffer (33,554,432 bytes, twice the 16 MiB the packet asks for)
/// grow the allocator peak by less than a quarter of one buffer - a donating append writes the
/// row in place, so the expected growth is one row (2 KiB) plus scratch. The control keeps a
/// `share()` of the target on the same cadence: the handle is never unique, no append
/// donates, every append copies the whole buffer, and the retire list holds the intermediates
/// of the eight-append window until the evaluate flushes them, so the control must grow by at
/// least two buffers, which proves the metric sees a copy (the F32 control of
/// `native_kv_ops.rs` proves the same through the retained `slice_update`).
#[test]
fn kv_append_donates_a_u32_buffer_over_200_appends() {
    let _serial = GPU_TEST.lock().unwrap();
    let buffer_bytes: u64 = 2 * 16384 * 256 * 4;
    assert!(
        buffer_bytes >= 16 * 1024 * 1024,
        "one buffer must be at least 16 MiB"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    let row = import(
        &mut kernels,
        &u32_pattern(&[1, 2, 1, 256], |index| {
            0x0bad_f00du32.wrapping_add(index as u32)
        }),
    );

    let previous_cap = kernels.set_retire_cap(8 << 30).unwrap();
    let cache_value = |index: usize| (index as u32).wrapping_mul(0x045d_9f3b);
    let row_value = |index: usize| 0x0bad_f00du32.wrapping_add(index as u32);

    // The donating loop: every append writes into the one donated buffer.
    let mut cache = import(&mut kernels, &u32_pattern(&[1, 2, 16384, 256], cache_value));
    kernels.reset_peak_memory().unwrap();
    let start = kernels.memory_stats().unwrap().active;
    let began = Instant::now();
    for step in 0..200u32 {
        cache = kernels.kv_append(cache, &row, &[0, 0, step, 0]).unwrap();
        if step % 8 == 7 {
            kernels.evaluate(&[&cache]).unwrap();
        }
    }
    let append_ms = began.elapsed().as_secs_f64() / 200.0 * 1e3;
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
    println!(
        "U32 kv_append: {append_ms:.3} ms per append, peak growth {growth} bytes over a \
         {buffer_bytes}-byte buffer"
    );
    assert!(
        growth < buffer_bytes / 4,
        "a donating append grew the peak by {growth} bytes; the bound is one quarter of a \
         buffer ({})",
        buffer_bytes / 4
    );

    // The written rows and one untouched row: element (g, t, d) sits at (g * 16384 + t) * 256 + d.
    let got = u32_values(&read(&mut kernels, &cache));
    for g in 0..2usize {
        for t in 0..=200usize {
            for d in 0..256usize {
                let expected = if t < 200 {
                    row_value(g * 256 + d)
                } else {
                    cache_value((g * 16384 + t) * 256 + d)
                };
                assert_eq!(
                    got[(g * 16384 + t) * 256 + d],
                    expected,
                    "head {g} row {t} lane {d} after 200 U32 appends"
                );
            }
        }
    }
    drop(cache);

    // The control: a kept `share()` blocks the donation, so every append copies the buffer.
    let mut retained = import(&mut kernels, &u32_pattern(&[1, 2, 16384, 256], cache_value));
    let held = retained.share();
    kernels.reset_peak_memory().unwrap();
    let start = kernels.memory_stats().unwrap().active;
    for step in 0..200u32 {
        retained = kernels.kv_append(retained, &row, &[0, 0, step, 0]).unwrap();
        if step % 8 == 7 {
            kernels.evaluate(&[&retained]).unwrap();
        }
    }
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
    println!("U32 kv_append with a kept share: peak growth {growth} bytes over the same buffer");
    assert!(
        growth >= buffer_bytes,
        "the kept-share control must show the copy: growth {growth} bytes, the bound is one \
         buffer ({}): a copying append holds the old and the new buffer at once",
        buffer_bytes
    );
    drop((held, retained, row));
    kernels.set_retire_cap(previous_cap).unwrap();
}

/// The decision 0032 fault hooks of the two new shims (packet KV8-OPS test e): `call`,
/// `null`, and `mismatch` on the first selected call of each. Every native failure is
/// terminal for the process, so each case runs in its own child process, as the rebind tests
/// of the adapter do.
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
        tensor::ExecutionTensor,
    };

    use super::{f32_pattern, import, u32_pattern};

    const CHILD: &str = "INFERENCE_KV8_FAULT_CHILD";
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

    /// One quantized attention call over a tiny pair: KVH 1, QH 2, D 64, cap 4, causal.
    fn prefix_call(
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
        kernels.quantized_attention_prefix(&operands[0], &keys, &values, 4, 0.125, true, 64, 8)
    }

    /// Builds the tiny operands: the query and the three residents of each side. They are built
    /// from imports only: the fault of these tests strikes the NEXT counted native operation,
    /// and a `quantize_rows` call here would take it before the call under test.
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
    fn quantize_rows_call_failure_is_terminal() {
        if delegated_to_child(
            "quantize_rows_call_failure_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import(
            &mut kernels,
            &f32_pattern(&[2, 128], |index| index as f32 / 8.0),
        );
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            kernels.quantize_rows(&x, 64, 8).map(|_| ()),
            "quantize rows",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantize_rows_null_output_is_terminal() {
        if delegated_to_child(
            "quantize_rows_null_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import(
            &mut kernels,
            &f32_pattern(&[2, 128], |index| index as f32 / 8.0),
        );
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            kernels.quantize_rows(&x, 64, 8).map(|_| ()),
            "quantize rows",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantize_rows_mismatch_is_terminal() {
        if delegated_to_child(
            "quantize_rows_mismatch_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import(
            &mut kernels,
            &f32_pattern(&[2, 128], |index| index as f32 / 8.0),
        );
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        // The widened codes no longer match the checked layout, so the metadata check fails.
        expect_backend(
            kernels.quantize_rows(&x, 64, 8).map(|_| ()),
            "quantize rows",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantized_attention_prefix_call_failure_is_terminal() {
        if delegated_to_child(
            "quantized_attention_prefix_call_failure_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            prefix_call(&mut kernels, &operands),
            "quantized attention prefix",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantized_attention_prefix_null_output_is_terminal() {
        if delegated_to_child(
            "quantized_attention_prefix_null_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            prefix_call(&mut kernels, &operands),
            "quantized attention prefix",
        );
        terminal_check(&mut kernels, &probe);
    }

    #[test]
    fn quantized_attention_prefix_mismatch_is_terminal() {
        if delegated_to_child(
            "quantized_attention_prefix_mismatch_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch"),
        ) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let operands = tiny_operands(&mut kernels);
        let probe = import(&mut kernels, &f32_pattern(&[4], |index| index as f32));
        expect_backend(
            prefix_call(&mut kernels, &operands),
            "quantized attention prefix",
        );
        terminal_check(&mut kernels, &probe);
    }
}
