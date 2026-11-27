//! The output head of a step: the final norm of one hidden row, the projection onto the
//! vocabulary and the logit cap.

use super::rows::{Norm, Rows, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::{DeviceBuffer, DeviceRange, NativeWeight};

/// The vocabulary matrix of a head.
#[derive(Clone, Copy)]
pub(crate) enum HeadWeight<'a> {
    /// A block-quantized matrix, read by the step's projections.
    Native(&'a NativeWeight),
    /// A BF16, F16 or F32 matrix of `rows` vocabulary entries by `cols`, read by the dense
    /// matvec on the f32 row.
    #[expect(dead_code, reason = "Inkling's head is its first reader")]
    Dense {
        codes: DeviceRange<'a>,
        dtype: DenseDtype,
        rows: usize,
        cols: usize,
    },
}

impl HeadWeight<'_> {
    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Native(w) => (w.n_rows(), w.k()),
            Self::Dense { rows, cols, .. } => (*rows, *cols),
        }
    }
}

/// The weights of the head and the rows it works in.
#[derive(Clone, Copy)]
pub(crate) struct Head<'a> {
    /// The final norm. `None`: the model hands the head a row it normed itself, which a dense
    /// matrix reads as it is.
    pub(crate) norm: Option<DeviceRange<'a>>,
    pub(crate) eps: f32,
    pub(crate) weight: HeadWeight<'a>,
    /// `cap * tanh(logit / cap)` on every logit, on the device after the projection.
    pub(crate) softcap: Option<f32>,
    /// The normed row, and its q8_1 row.
    pub(crate) normed: Rows<'a>,
    /// One float per vocabulary entry.
    pub(crate) logits: &'a DeviceBuffer,
}

impl Head<'_> {
    /// The logits of the hidden row `x`, queued on the step's stream.
    pub(crate) fn logits(&self, io: &StepIo<'_>, x: DeviceRange<'_>) -> Result<(), EngError> {
        let refuse = |what: &str| Err(EngError::BadRequest(format!("{}: {what}", io.family)));
        let (vocab, d) = self.weight.dims();
        let input = match self.norm {
            Some(weight) => {
                let Some(normed) = self.normed.f32 else {
                    return refuse("the head needs f32 rows for its norm");
                };
                let norm = Norm {
                    d,
                    eps: self.eps,
                    x,
                    delta: None,
                    weight: weight.slice(0, d * 4)?,
                };
                io.norm(
                    1,
                    norm,
                    normed.range(0, d * 4)?,
                    self.normed.q8.range(0, q8(d, 1))?,
                )?;
                normed.range(0, d * 4)?
            }
            None => x.slice(0, d * 4)?,
        };
        match self.weight {
            HeadWeight::Native(w) if self.norm.is_some() => {
                io.project(w, self.normed, self.logits, 1)?;
            }
            HeadWeight::Native(_) => {
                return refuse("a block-quantized head reads the q8_1 row of its own norm");
            }
            HeadWeight::Dense {
                codes,
                dtype,
                rows,
                cols,
            } => io.ctx.dense_gemv(
                io.stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, rows, cols),
                    dtype,
                    w: codes,
                    x: input,
                    y: self.logits.range(0, rows * 4)?,
                },
            )?,
        }
        match self.softcap {
            Some(cap) => {
                let logits = self.logits.range(0, vocab * 4)?;
                io.ctx.softcap_in_place(io.stream, logits, vocab, cap)
            }
            None => Ok(()),
        }
    }
}
