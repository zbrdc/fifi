//! Independent scalar layer fixture: zero Q/K, identity V/O, one raw-zero attention sink.

use super::{Expert, Fixture, Kind, Matrix};

pub(in crate::family::flint::cuda_ffn) const WIDTH: usize = 256;
pub(in crate::family::flint::cuda_ffn) const INTERMEDIATE: usize = 512;

fn matrix(rows: usize, cols: usize, factor: f32) -> Matrix {
    let mut values = vec![0.0; rows * cols];
    for row in 0..rows {
        values[row * cols + row % cols] = 1.0;
    }
    Matrix {
        rows,
        cols,
        values,
        factor,
    }
}

fn expert(down_factor: f32) -> Expert {
    Expert {
        gate: matrix(INTERMEDIATE, WIDTH, 1.0),
        up: matrix(INTERMEDIATE, WIDTH, 0.5),
        down: matrix(WIDTH, INTERMEDIATE, down_factor),
    }
}

pub(in crate::family::flint::cuda_ffn) fn weights() -> Fixture {
    Fixture {
        hidden: WIDTH,
        router: Matrix {
            rows: 8,
            cols: WIDTH,
            values: vec![0.0; 8 * WIDTH],
            factor: 1.0,
        },
        experts: (0..8).map(|id| expert(1.0 + id as f32 / 8.0)).collect(),
        shared: expert(0.25),
        dense: expert(1.0),
    }
}

fn normalize(input: &[f32]) -> Result<Vec<f32>, String> {
    let mut output = Vec::with_capacity(input.len());
    for row in input.as_chunks::<WIDTH>().0 {
        let squares = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
        let scale = (squares / WIDTH as f64 + 1e-6).sqrt().recip();
        for &value in row {
            output.push(super::bf16((f64::from(value) * scale) as f32)?);
        }
    }
    Ok(output)
}

fn attention(values: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    let mut sums = vec![0.0f64; WIDTH];
    let mut output = Vec::with_capacity(values.len());
    for token in 0..rows {
        for column in 0..WIDTH {
            sums[column] += f64::from(values[token * WIDTH + column]);
            // Zero Q/K gives unit key mass; the zero-valued sink adds one more unit.
            output.push(super::bf16((sums[column] / (token + 2) as f64) as f32)?);
        }
    }
    Ok(output)
}

pub(in crate::family::flint::cuda_ffn) fn run(
    fixture: &Fixture,
    input: &[f32],
    rows: usize,
    kind: Kind,
) -> Result<Vec<f32>, String> {
    if fixture.hidden != WIDTH || rows == 0 || rows.checked_mul(WIDTH) != Some(input.len()) {
        return Err("invalid first-light layer shape".into());
    }
    let input = input
        .iter()
        .map(|&x| super::bf16(x))
        .collect::<Result<Vec<_>, _>>()?;
    let projected = attention(&normalize(&input)?, rows)?;
    let residual = input
        .iter()
        .zip(projected)
        .map(|(&x, delta)| super::bf16(x + delta))
        .collect::<Result<Vec<_>, _>>()?;
    let ffn = super::run(fixture, &normalize(&residual)?, rows, kind)?;
    residual
        .iter()
        .zip(ffn.output)
        .map(|(&x, delta)| super::bf16(x + delta))
        .collect()
}

pub(in crate::family::flint::cuda_ffn) fn dense_then_moe(
    fixture: &Fixture,
    input: &[f32],
    rows: usize,
) -> Result<[Vec<f32>; 2], String> {
    let dense = run(fixture, input, rows, Kind::Dense)?;
    let moe = run(fixture, &dense, rows, Kind::Moe)?;
    Ok([dense, moe])
}

pub(in crate::family::flint::cuda_ffn) fn same_bf16(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual.is_finite()
                && actual.to_bits() & 0xffff == 0
                && actual.to_bits() == expected.to_bits()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_vector_comparison_preserves_signed_zero_and_publication() {
        assert!(same_bf16(&[0.0, 1.0, -2.0], &[0.0, 1.0, -2.0]));
        assert!(!same_bf16(&[-0.0], &[0.0]));
        assert!(!same_bf16(&[0.0], &[-0.0]));
        assert!(!same_bf16(&[1.0], &[]));
        assert!(!same_bf16(&[], &[1.0]));
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1.0f32.to_bits() + 1),
        ] {
            assert!(!same_bf16(&[value], &[value]));
        }
    }

    #[test]
    fn dense_whole_layer_matches_hand_calculated_publications() {
        let mut input = vec![0.0; WIDTH];
        input[0] = 1.0;
        let actual = run(&weights(), &input, 1, Kind::Dense).unwrap();
        let mut expected = vec![0.0; WIDTH];
        // BF16 sites: 16 norm -> 8 attention (one sink) -> 9 residual -> 16 norm -> 128 FFN.
        expected[0] = 137.0;
        assert!(same_bf16(&actual, &expected), "{actual:?} != {expected:?}");
    }

    #[test]
    fn dense_then_moe_publishes_each_layer_boundary() {
        let mut input = vec![0.0; WIDTH];
        input[0] = 1.0;
        let actual = dense_then_moe(&weights(), &input, 1).unwrap();
        let mut expected = [vec![0.0; WIDTH], vec![0.0; WIDTH]];
        expected[0][0] = 137.0;
        // Layer two: norm16 -> attention8 -> residual145 -> FFN200 -> BF16(345)=344.
        expected[1][0] = 344.0;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                same_bf16(actual, expected),
                "layer {index}: {actual:?} != {expected:?}"
            );
        }
        assert!(dense_then_moe(&weights(), &[], 0).is_err());
    }

    #[test]
    fn moe_whole_layer_matches_hand_calculated_publications() {
        let mut input = vec![0.0; WIDTH];
        input[0] = 1.0;
        let actual = run(&weights(), &input, 1, Kind::Moe).unwrap();
        let mut expected = vec![0.0; WIDTH];
        // Six published products sum to BF16 168; shared 32 and residual 9 give 209.
        expected[0] = 209.0;
        assert!(same_bf16(&actual, &expected), "{actual:?} != {expected:?}");
    }
}
