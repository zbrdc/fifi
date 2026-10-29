//! Weights in the GGUF file's own block layout, resident on the device:
//! [`CudaContext::upload_native`] reads a Q4_K, Q5_0, Q6_K or Q8_0 tensor through a
//! [`BlockReader`] in chunks of WHOLE ROWS and uploads each chunk at its byte offset into ONE
//! device allocation made up front. No repack runs: the bytes land as the file holds them,
//! row-major blocks, and the MMVQ kernels (`matvec_q`, `matvec_expert`) and the native
//! embedding gather read them there.
//!
//! A chunk holds as many rows as fit `chunk_limit` bytes and at least one, so a row wider than
//! the limit is its own chunk and the last chunk may be short; host memory stays at one chunk.
//! Rows may be ANY count from one up: the 8-row group rule is a tile rule, not a property of
//! the file. The fp16 scale scans of the tiled path do not apply (a native block keeps its
//! exact scales; the scans exist for tile dequantisation), so `upload_native` runs
//! none. A failed read or upload drops the device owner, as a failed tiled upload does.
//! Cleanup is skipped after terminal failure. Admission behaviour is the tiled path's.

use crate::EngError;
use crate::kernels::cuda::native;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::weights::GgufType;

use super::{BlockReader, CudaContext, DeviceBuffer, QuantTensor, device_error, host_source};

/// A quantized weight in the GGUF file's own block layout, resident on the device: one
/// allocation of `n_rows` rows of `k / block_size` blocks each, `experts` the stack depth of a
/// rank-3 tensor (`None` for a plain matrix). The `matvec_q`/`matvec_expert` kernels and the
/// native `embed` read it; the tiled readers refuse it with a typed error.
pub(crate) struct NativeWeight {
    buffer: DeviceBuffer,
    quant: QuantType,
    k: usize,
    n_rows: usize,
    experts: Option<usize>,
}

/// How a native upload splits into chunks of whole rows: `per_chunk` rows, at least one and as
/// many as fit `chunk_limit` bytes of source. No chunk splits a row; the last chunk may be
/// short.
#[derive(Clone, Copy)]
struct RowChunkPlan {
    /// Rows of the tensor.
    rows: usize,
    /// Rows per chunk.
    per_chunk: usize,
}

impl RowChunkPlan {
    fn new(rows: usize, row_bytes: usize, chunk_limit: usize) -> Self {
        Self {
            rows,
            per_chunk: (chunk_limit / row_bytes).clamp(1, rows),
        }
    }

    /// Rows of the chunk that starts at row `first` (at least one).
    fn count_of(&self, first: usize) -> usize {
        self.per_chunk.min(self.rows - first)
    }
}

impl CudaContext {
    /// Reads a Q4_K, Q5_0, Q6_K or Q8_0 tensor through `read` in chunks of whole rows and
    /// uploads the file's own blocks into one device allocation made up front.
    ///
    /// A chunk holds as many whole rows as fit `chunk_limit` bytes and at least one, however
    /// wide the row; the last chunk may be short. Returns `Checkpoint` naming the tensor for
    /// another type, a `k` off the type's block grid or above the matvec limit, a row count of
    /// zero, or a wrong source length; `MemoryPlan` when the device cannot hold the blocks or
    /// the host the chunk. An error from `read` is returned unchanged. A failed read or
    /// upload drops the device owner. Cleanup is skipped after terminal failure.
    pub(crate) fn upload_native(
        &self,
        tensor: QuantTensor<'_>,
        chunk_limit: usize,
        read: BlockReader<'_>,
    ) -> Result<NativeWeight, EngError> {
        self.ensure_usable()?;
        super::graph::ensure_closed("a native upload")?;
        let quant = checked_native(tensor)?;
        let name = tensor.name;
        // Unreachable in practice (`checked_native` proved k on the block grid), still a
        // typed error: the load path never panics.
        let row_bytes = quant.weight_bytes(1, tensor.k).ok_or_else(|| {
            EngError::Checkpoint(format!(
                "tensor {name} has {} values, which the native block grid cannot hold",
                tensor.k
            ))
        })?;
        let plan = RowChunkPlan::new(tensor.n_rows, row_bytes, chunk_limit);
        let buffer = self.alloc(tensor.source_len)?;
        let raw = self.owned(&buffer)?;
        let mut source = host_source(name, plan.per_chunk * row_bytes)?;
        for first in (0..plan.rows).step_by(plan.per_chunk) {
            let part = &mut source[..plan.count_of(first) * row_bytes];
            read(first * row_bytes, part)?;
            let _dispatch = self.enter()?;
            native::upload_at(raw, first * row_bytes, part).map_err(device_error)?;
        }
        Ok(NativeWeight {
            buffer,
            quant,
            k: tensor.k,
            n_rows: tensor.n_rows,
            experts: None,
        })
    }
}

impl NativeWeight {
    /// Rows (output values per token) of the weight: the TOTAL row count, which for a
    /// rank-3 stack is `experts` consecutive matrices of `n_rows / experts` rows each
    /// (the depth rides `experts()`).
    pub(crate) fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Values per row (input values per token) of the weight.
    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// The stack depth of a rank-3 tensor, `None` for a plain matrix. Read by the resident
    /// step's native stack lookups.
    pub(crate) fn experts(&self) -> Option<usize> {
        self.experts
    }

    /// The quantized format of the blocks.
    pub(crate) fn quant(&self) -> QuantType {
        self.quant
    }

    /// The device blocks, for launches and parity checks.
    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    /// Records the stack depth of a rank-3 tensor (`None` for a plain matrix). The upload
    /// reads only the tensor's flattened shape, so the loader writes the depth it read from
    /// the GGUF dims here.
    pub(crate) fn with_experts(mut self, experts: Option<usize>) -> Self {
        self.experts = experts;
        self
    }
}

/// Checks the tensor of a native upload: the type must be one of the four the MMVQ kernels
/// read, `k` a positive multiple of the type's block size up to `native::MAX_DIM`, `n_rows` at
/// least one (any count: no 8-row rule), and the source length exactly the native byte count.
fn checked_native(tensor: QuantTensor<'_>) -> Result<QuantType, EngError> {
    let quant = match tensor.ty {
        GgufType::Q4K => QuantType::Q4K,
        GgufType::Q50 => QuantType::Q50,
        GgufType::Q6K => QuantType::Q6K,
        GgufType::Q80 => QuantType::Q80,
        other => {
            return Err(EngError::Checkpoint(format!(
                "tensor {} has type {other:?}, which the CUDA native weights do not take",
                tensor.name
            )));
        }
    };
    let shape = |why: &str| {
        EngError::Checkpoint(format!(
            "tensor {} has {} rows of {} values, which the native layout cannot hold: {why}",
            tensor.name, tensor.n_rows, tensor.k
        ))
    };
    if tensor.n_rows == 0 {
        return Err(shape("the row count is zero"));
    }
    if tensor.k == 0 || !tensor.k.is_multiple_of(quant.block_size()) {
        return Err(shape("k is off the type's block grid"));
    }
    if tensor.k > native::MAX_DIM {
        return Err(shape("k is above the matvec limit"));
    }
    if quant.weight_bytes(tensor.n_rows, tensor.k) != Some(tensor.source_len) {
        return Err(EngError::Checkpoint(format!(
            "tensor {} has {} bytes of blocks for {} rows of {} values",
            tensor.name, tensor.source_len, tensor.n_rows, tensor.k
        )));
    }
    Ok(quant)
}

#[cfg(test)]
mod tests;
