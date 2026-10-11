//! Projection seam for host-orchestrated family forward passes (Flash-Next first).
//!
//! The forward blocks keep activations as host `f32` vectors. Every quantized
//! projection goes through [`Projector`]. The Mac adapter executes packed weights
//! through the MLX kernels. The test adapter multiplies dense `f32` matrices so
//! the block math runs in portable tests. Decision 0035 records this seam.

use crate::EngError;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod cuda;

/// Executes the quantized projections of one forward step.
///
/// All matrices use the checkpoint layout `[out, in]`, and every result is
/// `y = x · Wᵀ`. An adapter returns `BadRequest` when `x` does not match the
/// weight's input width, or when an expert or row index is out of range.
pub(crate) trait Projector {
    /// The adapter's handle for one stored weight or stacked weight.
    type Weight;

    /// Returns `x · Wᵀ` for a weight with logical shape `[out, in]`.
    fn linear(&mut self, weight: &Self::Weight, x: &[f32]) -> Result<Vec<f32>, EngError>;

    /// Returns `x · W[expert]ᵀ` for a stacked weight with logical shape `[E, out, in]`.
    fn expert_linear(
        &mut self,
        stack: &Self::Weight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError>;

    /// Returns the selected rows of a table with logical shape `[rows, cols]`.
    ///
    /// The result holds `rows.len() * cols` values in request order.
    fn rows(&mut self, table: &Self::Weight, rows: &[u32]) -> Result<Vec<f32>, EngError>;
}

/// A dense row-major matrix or matrix stack for the test adapter.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct DenseWeight {
    /// Stack depth. A plain matrix has depth 1.
    pub(crate) depth: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f32>,
}

#[cfg(test)]
impl DenseWeight {
    /// Builds a `[rows, cols]` matrix. Panics in tests when the length is wrong.
    pub(crate) fn matrix(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "dense matrix length");
        Self {
            depth: 1,
            rows,
            cols,
            data,
        }
    }

    /// Builds a `[depth, rows, cols]` stack. Panics in tests when the length is wrong.
    pub(crate) fn stack(depth: usize, rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), depth * rows * cols, "dense stack length");
        Self {
            depth,
            rows,
            cols,
            data,
        }
    }

    fn matvec(&self, offset: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if x.len() != self.cols {
            return Err(EngError::BadRequest(format!(
                "dense projection input has {} values, expected {}",
                x.len(),
                self.cols
            )));
        }
        let block = &self.data[offset..offset + self.rows * self.cols];
        Ok(block
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect())
    }
}

/// Test adapter that multiplies dense matrices and counts calls.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct DenseProjector {
    pub(crate) calls: usize,
}

#[cfg(test)]
impl Projector for DenseProjector {
    type Weight = DenseWeight;

    fn linear(&mut self, weight: &DenseWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.calls += 1;
        if weight.depth != 1 {
            return Err(EngError::BadRequest("linear needs a plain matrix".into()));
        }
        weight.matvec(0, x)
    }

    fn expert_linear(
        &mut self,
        stack: &DenseWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.calls += 1;
        let expert = expert as usize;
        if expert >= stack.depth {
            return Err(EngError::BadRequest(format!(
                "expert {expert} is outside stack depth {}",
                stack.depth
            )));
        }
        stack.matvec(expert * stack.rows * stack.cols, x)
    }

    fn rows(&mut self, table: &DenseWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        self.calls += 1;
        if table.depth != 1 {
            return Err(EngError::BadRequest("rows needs a plain matrix".into()));
        }
        let mut out = Vec::with_capacity(rows.len() * table.cols);
        for &row in rows {
            let row = row as usize;
            if row >= table.rows {
                return Err(EngError::BadRequest(format!(
                    "row {row} is outside table rows {}",
                    table.rows
                )));
            }
            out.extend_from_slice(&table.data[row * table.cols..(row + 1) * table.cols]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseProjector, DenseWeight, Projector};
    use crate::EngError;

    #[test]
    fn linear_multiplies_by_the_transposed_matrix() {
        // W = [[1, 2], [3, 4], [5, 6]] has shape [out 3, in 2]. x = [1, -1].
        let w = DenseWeight::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut p = DenseProjector::default();
        assert_eq!(p.linear(&w, &[1.0, -1.0]).unwrap(), [-1.0, -1.0, -1.0]);
        assert!(matches!(p.linear(&w, &[1.0]), Err(EngError::BadRequest(_))));
    }

    #[test]
    fn expert_linear_selects_one_matrix_of_the_stack() {
        // Expert 0 is the identity. Expert 1 swaps the two inputs.
        let w = DenseWeight::stack(2, 2, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let mut p = DenseProjector::default();
        assert_eq!(p.expert_linear(&w, 0, &[2.0, 3.0]).unwrap(), [2.0, 3.0]);
        assert_eq!(p.expert_linear(&w, 1, &[2.0, 3.0]).unwrap(), [3.0, 2.0]);
        assert!(matches!(
            p.expert_linear(&w, 2, &[2.0, 3.0]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(p.calls, 3);
    }

    #[test]
    fn rows_returns_rows_in_request_order() {
        let w = DenseWeight::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut p = DenseProjector::default();
        assert_eq!(p.rows(&w, &[2, 0]).unwrap(), [5.0, 6.0, 1.0, 2.0]);
        assert!(matches!(p.rows(&w, &[3]), Err(EngError::BadRequest(_))));
    }
}
