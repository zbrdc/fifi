//! Projection seam for host-orchestrated family forward passes (Flash-Next first).
//!
//! The forward blocks keep activations as host `f32` vectors. Every quantized
//! projection goes through [`Projector`]. The Mac adapter executes packed weights
//! through the MLX kernels. The test adapter multiplies dense `f32` matrices so
//! the block math runs in portable tests.

use crate::EngError;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod cuda;

/// Executes the quantized projections of one forward step.
///
/// All matrices use the checkpoint layout `[out, in]`, and every result is
/// `y = x · Wᵀ`. An adapter returns `BadRequest` when `x` does not match the
/// weight's input width, or when an expert or row index is out of range.
pub(crate) trait Projector {
    /// Projects one packed NVFP4 matrix against one activation row.
    ///
    /// The default refuses: only a backend that keeps the packed bytes on its
    /// Projects one packed NVFP4 matrix against one activation row.
    ///
    /// The default refuses: only a backend that keeps the packed bytes on its
    /// device answers this without expanding the matrix on the host.
    fn nvfp4_matvec(
        &mut self,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _ = (packed, scale, weight_scale_2, shape, input);
        Err(EngError::BadRequest(
            "packed NVFP4 projection is unavailable".to_string(),
        ))
    }

    /// Projects one packed NVFP4 matrix, keyed by a caller-assigned stable
    /// identity for the underlying checkpoint bytes.
    ///
    /// The default ignores the key and forwards to [`Projector::nvfp4_matvec`];
    /// a device-resident backend overrides this to reuse an admitted copy when
    /// the same key recurs. Keys must be stable for the whole run and derived
    /// only from immutable checkpoint coordinates (layer, expert, projection).
    fn nvfp4_matvec_keyed(
        &mut self,
        key: u64,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _ = key;
        self.nvfp4_matvec(packed, scale, weight_scale_2, shape, input)
    }

    /// Projects one stored BF16 matrix against one activation row, widening
    /// the 16-bit codes on the device so no F32 copy of the weight exists on
    /// the host or the device.
    ///
    /// The default refuses: only a backend with a BF16-input kernel answers.
    fn bf16_matvec(
        &mut self,
        codes: &[u8],
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _ = (codes, shape, input);
        Err(EngError::BadRequest(
            "stored BF16 projection is unavailable".to_string(),
        ))
    }

    /// Projects one stored F8_E4M3 matrix with its per-tensor scale against
    /// one activation row, decoding on the device so no F32 copy of the
    /// weight exists on the host or the device.
    ///
    /// The default refuses: only a backend with an F8-input kernel answers.
    fn f8_matvec(
        &mut self,
        codes: &[u8],
        weight_scale: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _ = (codes, weight_scale, shape, input);
        Err(EngError::BadRequest(
            "stored F8 projection is unavailable".to_string(),
        ))
    }

    /// The adapter's handle for one stored weight or stacked weight.
    type Weight;

    /// Returns `x · Wᵀ` for a weight with logical shape `[out, in]`.
    fn linear(&mut self, weight: &Self::Weight, x: &[f32]) -> Result<Vec<f32>, EngError>;

    /// Name-based compute on a device-resident weight (the handle path
    /// of the qwen36 CudaWeights pattern). The default refuses: only a
    /// backend that keeps the weight on its device under the given name
    /// answers this. Returns `None` when the name is not resident; the
    /// caller falls back to `linear(&Self::Weight, ..)`.
    fn linear_by_name(&mut self, _name: &str, _x: &[f32]) -> Option<Result<Vec<f32>, EngError>> {
        None
    }

    /// The stored shape of a device-resident weight, by name. The default
    /// returns None; only a backend with resident weights answers.
    fn resident_shape(&self, _name: &str) -> Option<[usize; 2]> {
        None
    }

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

#[cfg(test)]
mod keyed_default_tests {
    use super::*;

    struct Forward;

    impl Projector for Forward {
        type Weight = DenseWeight;

        fn linear(&mut self, _: &DenseWeight, _: &[f32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("unused".to_string()))
        }

        fn expert_linear(
            &mut self,
            _: &DenseWeight,
            _: u32,
            _: &[f32],
        ) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("unused".to_string()))
        }

        fn rows(&mut self, _: &DenseWeight, _: &[u32]) -> Result<Vec<f32>, EngError> {
            Err(EngError::BadRequest("unused".to_string()))
        }

        fn nvfp4_matvec(
            &mut self,
            packed: &[u8],
            scale: &[u8],
            weight_scale_2: f32,
            shape: [u32; 2],
            input: &[f32],
        ) -> Result<Vec<f32>, EngError> {
            Ok(vec![
                packed.len() as f32
                    + scale.len() as f32
                    + weight_scale_2
                    + shape[0] as f32
                    + shape[1] as f32
                    + input.len() as f32,
            ])
        }
    }

    #[test]
    fn the_keyed_default_forwards_to_the_unkeyed_call() {
        let mut projector = Forward;
        assert_eq!(
            projector
                .nvfp4_matvec_keyed(7, &[1, 2], &[3], 0.5, [2, 2], &[1.0, 2.0])
                .unwrap(),
            projector
                .nvfp4_matvec(&[1, 2], &[3], 0.5, [2, 2], &[1.0, 2.0])
                .unwrap()
        );
    }
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
