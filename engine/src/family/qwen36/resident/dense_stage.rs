//! One dense feed-forward block read into shared Metal pages on demand.

use super::{
    layout::{widen_layout, widen_stack_scales},
    moe::dense_rows,
    ops::{Ctx, short_import},
    weights::RDense,
};
use crate::{
    EngError,
    family::{
        qwen36::host::HostShape,
        stored::{RoleRange, SlicedWeight, StackLayout, read_matrix_into},
    },
    kernels::{HostPages, PackedPages, PageRange},
    tensor::{Dtype, ExecutionTensor},
    weights::StreamedCheckpoint,
};

pub(super) struct DenseStage {
    projections: Vec<Projection>,
    mapped_bytes: usize,
}

struct Projection {
    source: SlicedWeight,
    layout: StackLayout,
    narrow_read: Option<StackLayout>,
}

impl DenseStage {
    pub(super) fn new(
        checkpoint: &StreamedCheckpoint,
        weights: [SlicedWeight; 3],
        f32_scales: bool,
    ) -> Result<Self, EngError> {
        let mut projections = Vec::with_capacity(weights.len());
        let mut mapped_bytes = 0;
        let page = HostPages::page_size();
        for source in weights {
            let stored = StackLayout::matrix(checkpoint, &source, mapped_bytes, page)?;
            let expected = if f32_scales { Dtype::F32 } else { Dtype::Bf16 };
            let (layout, narrow_read) = if f32_scales && stored.scales.dtype == Dtype::Bf16 {
                let (wide, read) = widen_layout(&stored, page)?;
                (wide, Some(read))
            } else {
                (stored, None)
            };
            if layout.scales.dtype != expected
                || layout
                    .biases
                    .as_ref()
                    .is_some_and(|bias| bias.dtype != expected)
            {
                return Err(EngError::Checkpoint(format!(
                    "dense MLX staging needs {expected:?} scales and biases"
                )));
            }
            mapped_bytes = layout.end;
            projections.push(Projection {
                source,
                layout,
                narrow_read,
            });
        }
        Ok(Self {
            projections,
            mapped_bytes,
        })
    }

    /// Reserve the mapping plus a possible Metal fallback copy.
    pub(super) fn charge(&self) -> Result<u64, EngError> {
        (self.mapped_bytes as u64)
            .checked_mul(2)
            .ok_or_else(|| EngError::MemoryPlan("dense FFN staging bytes overflow".into()))
    }

    pub(super) fn evaluate(
        &self,
        ctx: &mut Ctx<'_>,
        checkpoint: &StreamedCheckpoint,
        shape: &HostShape,
        input: &ExecutionTensor,
        norm: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let mut pages = HostPages::try_new(self.mapped_bytes)?;
        let bytes = pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "dense FFN staging",
            message: "new staging pages are not writable".into(),
        })?;
        for projection in &self.projections {
            read_matrix_into(
                checkpoint,
                &projection.source,
                bytes,
                projection
                    .narrow_read
                    .as_ref()
                    .unwrap_or(&projection.layout),
            )?;
            if projection.narrow_read.is_some() {
                widen_stack_scales(bytes, &projection.layout)?;
            }
        }
        let wrapped = self
            .projections
            .iter()
            .map(|projection| {
                let layout = &projection.layout;
                Ok(PackedPages {
                    payload: matrix_range(&layout.payload)?,
                    scales: matrix_range(&layout.scales)?,
                    biases: layout.biases.as_ref().map(matrix_range).transpose()?,
                    logical_shape: matrix_shape(&layout.logical_shape)?,
                    quantization: layout.spec,
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        let (weights, plain) = ctx.kernels.import_packed_pages(&pages, &wrapped, &[])?;
        if !plain.is_empty() {
            return Err(short_import());
        }
        let [gate, up, down] = weights.try_into().map_err(|_| short_import())?;
        let weights = RDense { gate, up, down };
        let output = dense_rows(ctx, shape, &weights, input, norm)?;
        ctx.kernels.evaluate(&[&output])?;
        drop(weights);
        pages.release();
        ctx.kernels.evaluate(&[&output])?;
        if pages.in_use() {
            return Err(EngError::Backend {
                operation: "dense FFN staging",
                message: "dense FFN pages did not retire".into(),
            });
        }
        Ok(output)
    }
}

fn matrix_shape(shape: &[u32]) -> Result<Vec<u32>, EngError> {
    match shape {
        &[rows, 1, cols] => Ok(vec![rows, cols]),
        _ => Err(EngError::Checkpoint("invalid dense FFN page shape".into())),
    }
}

fn matrix_range(range: &RoleRange) -> Result<PageRange, EngError> {
    Ok(PageRange {
        offset: range.offset,
        dtype: range.dtype,
        shape: matrix_shape(&range.shape)?,
    })
}
