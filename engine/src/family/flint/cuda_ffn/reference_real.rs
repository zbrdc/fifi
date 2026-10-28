//! Independent real-layer scalar preparation; no backend operand-loss imitation.

use super::{Expected, Matrix, RealLayer, bf16, validate_matrix};
use crate::family::flint::reference::{AttentionPartial, ChunkWindow};
use ffn::project;

const HEAD_DIM: usize = 128;
const HIDDEN: usize = 2048;
const HEADS: usize = 32;
const KV_HEADS: usize = 4;

fn normalize(input: &[f32], gamma: &[f32]) -> Result<Vec<f32>, String> {
    if gamma.is_empty()
        || input.is_empty()
        || !input.len().is_multiple_of(gamma.len())
        || input.iter().chain(gamma).any(|value| !value.is_finite())
    {
        return Err("invalid normalization geometry or values".into());
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(gamma.len()) {
        let squares = row
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let inverse = (squares / gamma.len() as f64 + f64::from(1e-6_f32))
            .sqrt()
            .recip();
        for (&value, &weight) in row.iter().zip(gamma) {
            output.push(bf16(
                (f64::from(value) * inverse * f64::from(weight)) as f32,
            )?);
        }
    }
    Ok(output)
}

fn rotate(input: &mut [f32], heads: usize, pos0: u32) -> Result<(), String> {
    let width = heads
        .checked_mul(HEAD_DIM)
        .filter(|&width| width > 0)
        .ok_or_else(|| "invalid rotation head count".to_string())?;
    if input.is_empty()
        || !input.len().is_multiple_of(width)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("invalid rotation geometry or values".into());
    }
    let last_row = u32::try_from(input.len() / width - 1)
        .map_err(|_| "rotation row count exceeds u32".to_string())?;
    pos0.checked_add(last_row)
        .ok_or("rotation position overflow")?;
    for (offset, row) in input.chunks_exact_mut(width).enumerate() {
        let position = f64::from(pos0) + offset as f64;
        for head in row.as_chunks_mut::<HEAD_DIM>().0 {
            for pair in 0..HEAD_DIM / 2 {
                let angle = position * 10_000_f64.powf(-2.0 * pair as f64 / HEAD_DIM as f64);
                let (sine, cosine) = angle.sin_cos();
                let a = f64::from(head[pair]);
                let b = f64::from(head[pair + HEAD_DIM / 2]);
                head[pair] = bf16((a * cosine - b * sine) as f32)?;
                head[pair + HEAD_DIM / 2] = bf16((a * sine + b * cosine) as f32)?;
            }
        }
    }
    Ok(())
}

fn attention(qkv: [&[f32]; 3], sinks: &[f32], sliding: bool) -> Result<Vec<f32>, String> {
    let [q, k, v] = qkv;
    let rows = q.len() / (HEADS * HEAD_DIM);
    if rows == 0
        || !q.len().is_multiple_of(HEADS * HEAD_DIM)
        || k.len() != rows * KV_HEADS * HEAD_DIM
        || v.len() != k.len()
        || sinks.len() != HEADS
        || [q, k, v, sinks]
            .into_iter()
            .flatten()
            .any(|x| !x.is_finite())
    {
        return Err("invalid attention geometry or values".into());
    }
    let window = ChunkWindow::new(if sliding { 513 } else { rows }, 0, rows)
        .map_err(|error| format!("{error:?}"))?;
    let scale = f64::from((HEAD_DIM as f32).sqrt().recip());
    let mut output = Vec::with_capacity(q.len());
    for token in 0..rows {
        for (head, &sink) in sinks.iter().enumerate() {
            let query = &q[(token * HEADS + head) * HEAD_DIM..][..HEAD_DIM];
            let mut partial =
                AttentionPartial::new(HEAD_DIM).map_err(|error| format!("{error:?}"))?;
            for key in window
                .query_range(token)
                .map_err(|error| format!("{error:?}"))?
            {
                let start = (key * KV_HEADS + head / (HEADS / KV_HEADS)) * HEAD_DIM;
                let score = query
                    .iter()
                    .zip(&k[start..start + HEAD_DIM])
                    .map(|(&q, &k)| f64::from(q) * f64::from(k))
                    .sum::<f64>()
                    * scale;
                let values: Vec<_> = v[start..start + HEAD_DIM]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                partial
                    .push(score, &values)
                    .map_err(|error| format!("{error:?}"))?;
            }
            for value in partial
                .finish(f64::from(sink))
                .map_err(|error| format!("{error:?}"))?
            {
                output.push(bf16(value as f32)?);
            }
        }
    }
    Ok(output)
}

pub(in crate::family::flint::cuda_ffn) fn qk_producer_oracle(
    q: &[f32],
    k: &[f32],
    gamma: [&[f32]; 2],
    position: Option<u32>,
) -> Result<[Vec<f32>; 2], String> {
    let rows = q.len() / (HEADS * HEAD_DIM);
    if !(1..=4096).contains(&rows)
        || !q.len().is_multiple_of(HEADS * HEAD_DIM)
        || k.len() != rows * KV_HEADS * HEAD_DIM
        || gamma.iter().any(|weights| weights.len() != HEAD_DIM)
    {
        return Err("invalid D128 Q/K producer geometry".into());
    }
    let mut q = normalize(q, gamma[0])?;
    let mut k = normalize(k, gamma[1])?;
    if let Some(pos0) = position {
        rotate(&mut q, HEADS, pos0)?;
        rotate(&mut k, KV_HEADS, pos0)?;
    }
    Ok([q, k])
}

pub(in crate::family::flint::cuda_ffn) fn half_cache(input: &[f32]) -> Result<Vec<f32>, String> {
    input
        .iter()
        .map(|&value| {
            let stored = half::f16::from_f32(value).to_f32();
            if !value.is_finite() || value.to_bits() & 0xffff != 0 || !stored.is_finite() {
                return Err("invalid BF16 input or overflow in physical F16 cache".into());
            }
            Ok(stored)
        })
        .collect()
}

fn projection_rows(input: &[f32], matrix: &Matrix) -> Result<Vec<f32>, String> {
    let mut output = Vec::new();
    for row in input.chunks_exact(matrix.cols) {
        output.extend(project(row, matrix)?);
    }
    Ok(output)
}

fn add(input: &[f32], delta: &[f32]) -> Result<Vec<f32>, String> {
    if input.len() != delta.len() {
        return Err("residual geometry differs".into());
    }
    input
        .iter()
        .zip(delta)
        .map(|(&x, &d)| bf16(x + d))
        .collect()
}

fn validate(layer: &RealLayer<'_>, input: &[f32], rows: usize) -> Result<(), String> {
    let dimensions = [
        (layer.pre_norm.len(), HIDDEN),
        (layer.post_norm.len(), HIDDEN),
        (layer.q_norm.len(), HEAD_DIM),
        (layer.k_norm.len(), HEAD_DIM),
        (layer.sinks.len(), HEADS),
    ];
    if rows == 0
        || layer.ffn.hidden != HIDDEN
        || rows.checked_mul(HIDDEN) != Some(input.len())
        || dimensions
            .iter()
            .any(|&(actual, expected)| actual != expected)
        || layer.sinks.iter().any(|v| !v.is_finite())
    {
        return Err("invalid real-layer geometry or sinks".into());
    }
    for (matrix, shape) in [
        (layer.q, [HEADS * HEAD_DIM, HIDDEN]),
        (layer.k, [KV_HEADS * HEAD_DIM, HIDDEN]),
        (layer.v, [KV_HEADS * HEAD_DIM, HIDDEN]),
        (layer.o, [HIDDEN, HEADS * HEAD_DIM]),
    ] {
        validate_matrix(matrix)?;
        if [matrix.rows, matrix.cols] != shape {
            return Err("invalid real-layer projection shape".into());
        }
    }
    Ok(())
}

fn prefix(
    layer: &RealLayer<'_>,
    input: &[f32],
    rows: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    validate(layer, input, rows)?;
    let input = input
        .iter()
        .map(|&v| bf16(v))
        .collect::<Result<Vec<_>, _>>()?;
    let normalized = normalize(&input, layer.pre_norm)?;
    let mut q = normalize(&projection_rows(&normalized, layer.q)?, layer.q_norm)?;
    let mut k = normalize(&projection_rows(&normalized, layer.k)?, layer.k_norm)?;
    let v = projection_rows(&normalized, layer.v)?;
    if layer.sliding {
        rotate(&mut q, HEADS, 0)?;
        rotate(&mut k, KV_HEADS, 0)?;
    }
    let k = half_cache(&k)?;
    let v = half_cache(&v)?;
    let attended = attention([&q, &k, &v], layer.sinks, layer.sliding)?;
    let residual = add(&input, &projection_rows(&attended, layer.o)?)?;
    let normalized = normalize(&residual, layer.post_norm)?;
    Ok((residual, normalized))
}

/// Whole first chunk at absolute position zero, with the explicit opt-in physical F16 KV.
/// Other Expected fields retain the FFN routing trace; output is the final layer residual.
pub(in crate::family::flint::cuda_ffn) fn run(
    layer: &RealLayer<'_>,
    input: &[f32],
    rows: usize,
) -> Result<Expected, String> {
    let (residual, normalized) = prefix(layer, input, rows)?;
    let mut expected = ffn::run(layer.ffn, &normalized, rows, layer.kind)?;
    expected.output = add(&residual, &expected.output)?;
    Ok(expected)
}

#[path = "reference_real_ffn.rs"]
mod ffn;
#[path = "reference_gguf.rs"]
pub(in crate::family::flint::cuda_ffn) mod gguf;

#[cfg(test)]
mod tests {
    use super::super::layer::same_bf16;
    use super::*;

    #[test]
    fn nonzero_head_norm_then_rope_keeps_both_bf16_publications() {
        let mut input = vec![0.0; 2 * HEAD_DIM];
        input[0] = 1.0;
        input[HEAD_DIM] = 1.0;
        let mut actual = normalize(&input, &[1.0; HEAD_DIM]).unwrap();
        let mut expected = vec![0.0; 2 * HEAD_DIM];
        expected[0] = 11.3125;
        expected[HEAD_DIM] = 11.3125;
        assert!(same_bf16(&actual, &expected), "{actual:?} != {expected:?}");
        rotate(&mut actual, 1, 0).unwrap();
        // Position one rotates the already-published 11.3125, not the unrounded norm.
        expected[HEAD_DIM] = 6.125;
        expected[HEAD_DIM + HEAD_DIM / 2] = 9.5;
        assert!(same_bf16(&actual, &expected), "{actual:?} != {expected:?}");
    }

    #[test]
    fn causal_attention_maps_gqa8_and_adds_one_raw_sink() {
        let q = vec![0.0; 2 * 32 * HEAD_DIM];
        let k = vec![0.0; 2 * 4 * HEAD_DIM];
        let mut v = vec![0.0; k.len()];
        for (head, values) in v.as_chunks_mut::<HEAD_DIM>().0.iter_mut().enumerate() {
            values.fill((head % 4 + 1) as f32);
        }
        let mut expected = vec![0.0; q.len()];
        for (row, heads) in [[0.5, 1.0, 1.5, 2.0], [0.66796875, 1.3359375, 2.0, 2.671875]]
            .iter()
            .enumerate()
        {
            for head in 0..32 {
                let start = (row * 32 + head) * HEAD_DIM;
                expected[start..start + HEAD_DIM].fill(heads[head / 8]);
            }
        }
        for sliding in [false, true] {
            let actual = attention([&q, &k, &v], &[0.0; 32], sliding).unwrap();
            assert!(same_bf16(&actual, &expected));
        }
    }

    #[test]
    fn nonzero_attention_scores_keep_scale_separate_from_the_raw_sink() {
        let mut q = vec![0.0; 32 * HEAD_DIM];
        for head in q.as_chunks_mut::<HEAD_DIM>().0 {
            head[0] = 1.0;
        }
        let k = vec![1.0; 4 * HEAD_DIM];
        let v = k.clone();
        let score = f64::from((HEAD_DIM as f32).sqrt().recip());
        let mass = 1.0 / (1.0 + (f64::from(0.3_f32) - score).exp());
        let expected = vec![bf16(mass as f32).unwrap(); q.len()];
        let actual = attention([&q, &k, &v], &[0.3; 32], false).unwrap();
        assert!(same_bf16(&actual, &expected));
    }

    #[test]
    fn physical_f16_cache_preserves_signed_underflow_and_refuses_bad_inputs() {
        let tiny = bf16(1e-8).unwrap();
        assert!(same_bf16(
            &half_cache(&[tiny, -tiny]).unwrap(),
            &[0.0, -0.0]
        ));
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::from_bits(1.0_f32.to_bits() + 1),
            bf16(70_000.0).unwrap(),
        ] {
            assert!(half_cache(&[bad]).is_err());
        }
    }

    #[test]
    fn real_layer_refuses_incompatible_geometry_before_projection() {
        use super::super::{Expert, Fixture, Kind};
        let matrix = Matrix {
            rows: 1,
            cols: 1,
            values: vec![0.0],
            factor: 1.0,
        };
        let expert = Expert {
            gate: matrix.clone(),
            up: matrix.clone(),
            down: matrix.clone(),
        };
        let fixture = Fixture {
            hidden: 1,
            router: matrix.clone(),
            experts: vec![],
            shared: expert.clone(),
            dense: expert,
        };
        let layer = RealLayer {
            q: &matrix,
            k: &matrix,
            v: &matrix,
            o: &matrix,
            pre_norm: &[1.0],
            post_norm: &[1.0],
            q_norm: &[1.0],
            k_norm: &[1.0],
            sinks: &[0.0],
            ffn: &fixture,
            kind: Kind::Dense,
            sliding: false,
        };
        assert!(run(&layer, &[0.0], 1).is_err());
        assert!(run(&layer, &[], 0).is_err());
    }

    #[test]
    fn qk_producer_checks_rows_positions_and_uses_ordered_publications() {
        use crate::family::flint::cuda_ffn::qk_producer_oracle as oracle;
        let mut q = vec![0.0; HEADS * HEAD_DIM];
        let mut k = vec![0.0; KV_HEADS * HEAD_DIM];
        q[0] = 1.0;
        k[0] = 1.0;
        let gamma = [&[1.0; HEAD_DIM][..]; 2];
        let full = oracle(&q, &k, gamma, None).unwrap();
        assert_eq!(full[0][0], 11.3125);
        assert_eq!(full[1][0], 11.3125);
        let sliding = oracle(&q, &k, gamma, Some(1)).unwrap();
        assert_eq!([sliding[0][0], sliding[0][HEAD_DIM / 2]], [6.125, 9.5]);
        assert_eq!([sliding[1][0], sliding[1][HEAD_DIM / 2]], [6.125, 9.5]);
        assert!(oracle(&q, &k, gamma, Some(u32::MAX)).is_ok());
        assert!(oracle(&q.repeat(2), &k.repeat(2), gamma, Some(u32::MAX)).is_err());
        assert!(oracle(&[], &[], gamma, None).is_err());
        assert!(oracle(&q[..q.len() - 1], &k, gamma, None).is_err());
        assert!(oracle(&q, &k[..k.len() - 1], gamma, None).is_err());
        assert!(oracle(&q, &k, [&[]; 2], None).is_err());
        assert!(oracle(&q.repeat(4097), &k.repeat(4097), gamma, None).is_err());
        q[0] = f32::NAN;
        assert!(oracle(&q, &k, gamma, None).is_err());
    }

    #[test]
    fn normalization_and_rope_refuse_invalid_geometry_and_nonfinite_values() {
        assert!(normalize(&[], &[1.0; HEAD_DIM]).is_err());
        assert!(normalize(&[1.0], &[]).is_err());
        assert!(normalize(&[1.0], &[1.0; HEAD_DIM]).is_err());
        assert!(normalize(&[f32::NAN; HEAD_DIM], &[1.0; HEAD_DIM]).is_err());
        assert!(normalize(&[1.0; HEAD_DIM], &[f32::INFINITY; HEAD_DIM]).is_err());
        assert!(rotate(&mut [0.0; HEAD_DIM], 0, 0).is_err());
        assert!(rotate(&mut [0.0; HEAD_DIM], usize::MAX, 0).is_err());
        assert!(rotate(&mut [0.0; HEAD_DIM - 1], 1, 0).is_err());
        assert!(rotate(&mut [f32::NAN; HEAD_DIM], 1, 0).is_err());
    }
}
