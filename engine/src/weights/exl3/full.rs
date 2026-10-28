use half::f16;

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

use super::Matrix;

impl Matrix {
    /// Reconstructs ideal fp64 weights in [input, output] order, without bias.
    pub fn reconstruct_fp64_into(&self, output: &mut [f64]) -> Result<(), EngError> {
        let [k, n] = self.shape();
        let count = k
            .checked_mul(n)
            .ok_or_else(|| EngError::MemoryPlan("EXL3 fp64 extent overflow".into()))?;
        if output.len() != count {
            return Err(EngError::BadRequest(format!(
                "EXL3 fp64 output requires {count} elements"
            )));
        }
        let suh = host(self.suh())?;
        let svh = host(self.svh())?;
        let mut bits = Vec::new();
        bits.try_reserve_exact(count).map_err(|error| {
            EngError::MemoryPlan(format!("EXL3 fp16 reference buffer: {error}"))
        })?;
        bits.resize(count, 0u16);
        self.codebook()
            .reconstruct_into(self.trellis(), &mut bits)?;
        for (value, bits) in output.iter_mut().zip(bits) {
            *value = f16::from_bits(bits).to_f64();
        }
        for block in output.chunks_exact_mut(128 * n) {
            for column in 0..n {
                hadamard128(&mut block[column..], n);
            }
        }
        for (row, values) in output.chunks_exact_mut(n).enumerate() {
            let scale = half_at(suh, row);
            for value in values.iter_mut() {
                *value *= scale;
            }
            for block in values.as_chunks_mut::<128>().0 {
                hadamard128(block, 1);
            }
            for (column, value) in values.iter_mut().enumerate() {
                *value *= half_at(svh, column);
            }
        }
        Ok(())
    }

    /// Projects row-major [rows, input] batches in fp64, including optional bias.
    pub fn project_fp64_into(
        &self,
        input: &[f64],
        rows: usize,
        output: &mut [f64],
    ) -> Result<(), EngError> {
        let [k, n] = self.shape();
        let input_count = rows
            .checked_mul(k)
            .ok_or_else(|| EngError::MemoryPlan("EXL3 fp64 input span overflow".into()))?;
        let output_count = rows
            .checked_mul(n)
            .ok_or_else(|| EngError::MemoryPlan("EXL3 fp64 output span overflow".into()))?;
        if rows == 0 || input.len() != input_count || output.len() != output_count {
            return Err(EngError::BadRequest(
                "EXL3 fp64 projection has invalid batch spans".into(),
            ));
        }
        let count = k
            .checked_mul(n)
            .ok_or_else(|| EngError::MemoryPlan("EXL3 fp64 extent overflow".into()))?;
        let bias = self
            .bias()
            .map(|tensor| host(tensor).map(|bytes| (tensor.dtype(), bytes)))
            .transpose()?;
        // shortcut: materializes full W; replace only if oracle fixtures outgrow host memory.
        let mut weights = Vec::new();
        weights.try_reserve_exact(count).map_err(|error| {
            EngError::MemoryPlan(format!("EXL3 fp64 reference buffer: {error}"))
        })?;
        weights.resize(count, 0.0);
        self.reconstruct_fp64_into(&mut weights)?;
        for (x, y) in input.chunks_exact(k).zip(output.chunks_exact_mut(n)) {
            y.fill(0.0);
            for (x, w) in x.iter().zip(weights.chunks_exact(n)) {
                for (y, w) in y.iter_mut().zip(w) {
                    *y += x * w;
                }
            }
            // A bias added before reduction can disappear during finite cancellation.
            for (column, value) in y.iter_mut().enumerate() {
                *value += bias_at(bias, column)?;
            }
        }
        Ok(())
    }
}

fn host(tensor: &Tensor) -> Result<&[u8], EngError> {
    tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 fp64 reference requires host components".into()))
}

fn half_at(bytes: &[u8], index: usize) -> f64 {
    let offset = index * 2;
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f64()
}

fn bias_at(bias: Option<(Dtype, &[u8])>, index: usize) -> Result<f64, EngError> {
    match bias {
        None => Ok(0.0),
        Some((Dtype::F16, bytes)) => Ok(half_at(bytes, index)),
        Some((Dtype::F32, bytes)) => {
            let bytes = &bytes[index * 4..index * 4 + 4];
            Ok(f64::from(f32::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ])))
        }
        _ => Err(EngError::Checkpoint(
            "EXL3 fp64 bias must be F16 or F32".into(),
        )),
    }
}

fn hadamard128(values: &mut [f64], stride: usize) {
    for width in [1, 2, 4, 8, 16, 32, 64] {
        for block in (0..128).step_by(2 * width) {
            for lane in 0..width {
                let first = (block + lane) * stride;
                let second = (block + lane + width) * stride;
                let (a, b) = (values[first], values[second]);
                values[first] = a + b;
                values[second] = a - b;
            }
        }
    }
    let scale = 1.0 / 128.0f64.sqrt();
    for lane in 0..128 {
        values[lane * stride] *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::hadamard128;

    #[test]
    fn strided_hadamard_matches_sylvester_signs_and_is_orthogonal() {
        for stride in [1, 3] {
            let mut values = vec![42.0; 128 * stride];
            for lane in 0..128 {
                values[lane * stride] = if lane == 65 { -2.0 } else { 0.0 };
            }
            hadamard128(&mut values, stride);
            // H[i,65] = (-1)^popcount(i & 65) / sqrt(128).
            for (index, value) in values.iter().enumerate() {
                let expected = if index.is_multiple_of(stride) {
                    let sign = if ((index / stride) & 65usize).count_ones().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    -2.0 * sign / 128.0f64.sqrt()
                } else {
                    42.0
                };
                assert!(
                    (value - expected).abs() < 1e-12,
                    "stride={stride},index={index}"
                );
            }
            hadamard128(&mut values, stride);
            for (index, value) in values.iter().enumerate() {
                let expected = if index == 65 * stride {
                    -2.0
                } else if index.is_multiple_of(stride) {
                    0.0
                } else {
                    42.0
                };
                assert!(
                    (value - expected).abs() < 1e-12,
                    "roundtrip,stride={stride},index={index}"
                );
            }
        }
    }
}
