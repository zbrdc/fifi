//! Host projection adapter for portable NemotronH forwards.

use crate::{
    EngError,
    family::project::Projector,
    tensor::{Dtype, Tensor},
};

/// Executes dense F32 projections for the portable Mamba caller.
pub(crate) struct HostProjector;

impl Projector for HostProjector {
    fn nvfp4_matvec(
        &mut self,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let rows = shape[0] as usize;
        let columns = shape[1] as usize;
        let mut output = vec![0.0; rows];
        crate::kernels::nvfp4::WeightBlock {
            packed,
            weight_scale: scale,
            weight_scale_2,
            input_scale: None,
        }
        .matvec(rows, columns, input, &mut output)?;
        Ok(output)
    }

    type Weight = Tensor;

    fn linear(&mut self, weight: &Tensor, input: &[f32]) -> Result<Vec<f32>, EngError> {
        let &[rows, columns] = weight.shape() else {
            return Err(invalid("linear weight must be a matrix"));
        };
        if weight.dtype() != Dtype::F32 {
            return Err(invalid("linear weight must be F32"));
        }
        let rows = usize::try_from(rows).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(columns).map_err(|_| invalid("column count exceeds host limits"))?;
        if rows == 0 || columns == 0 {
            return Err(invalid("linear dimensions must be positive"));
        }
        if input.len() != columns {
            return Err(invalid("linear input length differs from the weight shape"));
        }
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("linear weight byte length exceeds host limits"))?;
        let bytes = weight
            .host_bytes()
            .ok_or_else(|| invalid("linear weight is not host-backed"))?;
        if bytes.len() != expected_bytes {
            return Err(invalid("linear weight byte length differs from its shape"));
        }
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes));
        let values: Vec<_> = values.collect();
        Ok(values
            .chunks_exact(columns)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect())
    }

    fn expert_linear(&mut self, _: &Tensor, _: u32, _: &[f32]) -> Result<Vec<f32>, EngError> {
        Err(invalid(
            "host Mamba projector does not support expert_linear",
        ))
    }

    fn rows(&mut self, _: &Tensor, _: &[u32]) -> Result<Vec<f32>, EngError> {
        Err(invalid("host Mamba projector does not support rows"))
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "one projection description; splitting would churn every call site"
)]
pub(crate) fn project_rows<P: Projector<Weight = Tensor>>(
    projector: &mut P,
    weight: &Tensor,
    input: &[f32],
    input_width: usize,
    output_width: usize,
    operation: &'static str,
    name: &str,
    weight_name: Option<&str>,
) -> Result<Vec<f32>, EngError> {
    if input_width == 0 || output_width == 0 || !input.len().is_multiple_of(input_width) {
        return Err(invalid("projection input does not match its row width"));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(invalid("projection input is not finite"));
    }
    let rows = input.len() / input_width;
    let capacity = rows
        .checked_mul(output_width)
        .filter(|elements| {
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| invalid("projection output exceeds host limits"))?;
    let mut output = Vec::with_capacity(capacity);
    for row in input.chunks_exact(input_width) {
        let projected = match weight_name.and_then(|wn| projector.linear_by_name(wn, row)) {
            Some(result) => result?,
            None => projector.linear(weight, row)?,
        };
        if projected.len() != output_width {
            return Err(EngError::Backend {
                operation,
                message: format!(
                    "{name} projection returned {} values, expected {output_width}",
                    projected.len()
                ),
            });
        }
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(invalid("projection output is not finite"));
        }
        output.extend(projected);
    }
    Ok(output)
}

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("NemotronH host projector: {message}"))
}

#[cfg(test)]
mod tests {
    use super::HostProjector;
    use crate::{
        EngError,
        family::project::Projector,
        tensor::{Dtype, Tensor},
    };

    fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn linear_multiplies_the_transposed_row_major_matrix() {
        let weight = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            HostProjector.linear(&weight, &[1.0, -2.0, 0.5]).unwrap(),
            [-1.5, -3.0]
        );
    }

    #[test]
    fn linear_rejects_malformed_rank_and_activation_width() {
        let matrix = tensor(&[2, 2], &[1.0; 4]);
        let vector = tensor(&[4], &[1.0; 4]);
        let mut projector = HostProjector;
        assert!(matches!(
            projector.linear(&vector, &[1.0; 4]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            projector.linear(&matrix, &[1.0]),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn host_projector_rejects_unsupported_operations() {
        let weight = tensor(&[1, 1], &[1.0]);
        let mut projector = HostProjector;
        assert!(matches!(
            projector.expert_linear(&weight, 0, &[1.0]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            projector.rows(&weight, &[0]),
            Err(EngError::BadRequest(_))
        ));
    }
}
