//! The token tail of a decode step: what turns a step over rows into a step over token ids.
//!
//! Before the visits, the embedding row of the token in the step parameters is gathered on the
//! device. After the head, the id of the largest logit goes back into the step parameters and
//! into eight bytes the host reads. A greedy token then costs the host one small read; the
//! logits row stays on the device for whoever wants it.
//!
//! The gather reads the table as the file stores it and writes the row a host gather of the same
//! table writes, so a step from a token id has the logits of a step from that host row. The
//! pick takes the lowest id among equal logits, which is the host sampler's greedy rule.

use crate::EngError;
use crate::family::mem::DEVICE;
use crate::kernels::cuda::resident::embed::EmbedTable;
use crate::kernels::cuda::resident::step_params::{
    ARGMAX_PART_BYTES, ArgmaxCall, OFFSET_STATUS, OFFSET_TOKEN_ID, RESULT_PAIR_BYTES, ResultPair,
    STEP_PARAMS_BYTES,
};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};
use crate::weights::GgufWeights;

/// The tensor `name` of `file` as an embedding table of `rows` by `k` on the device, when a
/// device gather reads its stored type and the device has the room; `None` leaves the gather to
/// the host.
pub(crate) fn device_table(
    ctx: &CudaContext,
    file: &GgufWeights,
    name: &str,
    (rows, k): (usize, usize),
) -> Result<Option<EmbedTable>, EngError> {
    let Some(tensor) = file.header().tensor(name) else {
        return Ok(None);
    };
    let (kind, len) = (tensor.ggml_type(), tensor.byte_len() as usize);
    let reserve = DEVICE.embedding_table_reserve as usize;
    if !EmbedTable::gathers(kind) || ctx.memory_info()?.free.saturating_sub(len) < reserve {
        return Ok(None);
    }
    let mut bytes = vec![0u8; len];
    file.read_range(name, 0, &mut bytes)?;
    EmbedTable::upload(ctx, name, (kind, rows, k), &bytes).map(Some)
}

/// The table and the scratch of the pick.
pub(crate) struct Tail {
    table: EmbedTable,
    part: DeviceBuffer,
    result: DeviceBuffer,
}

impl Tail {
    pub(crate) fn new(ctx: &CudaContext, table: EmbedTable) -> Result<Self, EngError> {
        Ok(Self {
            table,
            part: ctx.allocate(ARGMAX_PART_BYTES)?,
            result: ctx.allocate(RESULT_PAIR_BYTES)?,
        })
    }

    /// Queues the gather of the row of the token id in `params` into `row`. The caller has
    /// checked the id against the table: a bad id would flag the status word, which the first
    /// kernel of the visits resets.
    pub(crate) fn gather(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        params: &DeviceBuffer,
        row: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let words = [
            params.range(OFFSET_TOKEN_ID, 4)?,
            row,
            params.range(OFFSET_STATUS, 4)?,
        ];
        self.table.gather(ctx, stream, words, 1)
    }

    /// Queues the pick over the first `vocab` values of `logits`: the id goes into `params` and
    /// into the bytes [`Self::picked`] reads. It is the last kernel of the step that writes the
    /// status word.
    pub(crate) fn pick(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        logits: &DeviceBuffer,
        vocab: usize,
        params: &DeviceBuffer,
    ) -> Result<(), EngError> {
        ctx.argmax(
            stream,
            ArgmaxCall {
                n: vocab,
                logits: logits.range(0, vocab * 4)?,
                part: self.part.range(0, ARGMAX_PART_BYTES)?,
                params: params.range(0, STEP_PARAMS_BYTES)?,
                result: Some(self.result.range(0, RESULT_PAIR_BYTES)?),
            },
        )
    }

    /// Waits for the step and returns the id it picked; a flagged step is an error.
    pub(crate) fn picked(&self, stream: &DeviceStream) -> Result<u32, EngError> {
        let mut pair = [0u8; RESULT_PAIR_BYTES];
        self.result.read_at(stream, 0, &mut pair)?;
        let picked = ResultPair::from_bytes(pair).into_token()?;
        u32::try_from(picked).map_err(|_| EngError::Backend {
            operation: "token_pick",
            message: format!("the device picked the id {picked}"),
        })
    }
}
