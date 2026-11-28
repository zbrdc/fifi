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
use crate::family::sampler::Sampler;
use crate::kernels::cuda::resident::block_ops::BlockOp;
use crate::kernels::cuda::resident::embed::{EmbedTable, check_host_ids};
use crate::kernels::cuda::resident::shortlist::{self, ShortlistCall, ShortlistMassCall};
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

/// The table and the scratch of the pick (and optional chunk-gather scratch).
pub(crate) struct Tail {
    table: EmbedTable,
    part: DeviceBuffer,
    result: DeviceBuffer,
    chunk: Option<(DeviceBuffer, DeviceBuffer, usize)>,
}

impl Tail {
    pub(crate) fn new(ctx: &CudaContext, table: EmbedTable) -> Result<Self, EngError> {
        Ok(Self {
            table,
            part: ctx.allocate(ARGMAX_PART_BYTES)?,
            result: ctx.allocate(RESULT_PAIR_BYTES)?,
            chunk: None,
        })
    }

    /// Opens the embedding table `name` of `file` on the device when free memory admits it, with
    /// scratch for both single-token decode picks and prompt chunks up to `max_rows` tokens.
    pub(crate) fn open_chunks(
        ctx: &CudaContext,
        file: &GgufWeights,
        name: &str,
        (rows, k): (usize, usize),
        max_rows: usize,
    ) -> Result<Option<Self>, EngError> {
        let Some(table) = device_table(ctx, file, name, (rows, k))? else {
            return Ok(None);
        };
        let mut tail = Self::new(ctx, table)?;
        tail.chunk = Some((ctx.allocate(max_rows * 4)?, ctx.upload(&[0u8; 4])?, rows));
        Ok(Some(tail))
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

    /// Queues a single-token gather from `params` and broadcasts it across `streams` residual
    /// streams into `resid` (`[streams][hidden]`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather_step_streams(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        params: &DeviceBuffer,
        (streams, hidden): (usize, usize),
        scratch_rows: &DeviceBuffer,
        resid: &DeviceBuffer,
    ) -> Result<(), EngError> {
        if !(1..=16).contains(&streams) {
            return Err(EngError::BadRequest(format!(
                "tail gather_step_streams: invalid streams {streams}"
            )));
        }
        let dst = if streams == 1 { resid } else { scratch_rows };
        self.gather(ctx, stream, params, dst.range(0, hidden * 4)?)?;
        if streams > 1 {
            ctx.block_op(
                stream,
                BlockOp::hc_broadcast(
                    1,
                    streams,
                    hidden,
                    scratch_rows.range(0, hidden * 4)?,
                    resid.range(0, streams * hidden * 4)?,
                )?,
            )?;
        }
        Ok(())
    }

    /// Gathers a chunk of `tokens` on the device and broadcasts each row bit-for-bit across
    /// `streams` residual streams into `resid` (`[n][streams][hidden]`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather_chunk(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        tokens: &[u32],
        (streams, hidden): (usize, usize),
        scratch_rows: &DeviceBuffer,
        resid: &DeviceBuffer,
    ) -> Result<(), EngError> {
        if !(1..=16).contains(&streams) {
            return Err(EngError::BadRequest(format!(
                "tail gather_chunk: invalid streams {streams}"
            )));
        }
        let Some((ids_buf, status_buf, vocab)) = &self.chunk else {
            return Err(EngError::BadRequest(
                "tail gather_chunk: chunk scratch was not allocated".into(),
            ));
        };
        let n = tokens.len();
        let ids: Vec<i32> = tokens
            .iter()
            .map(|&t| i32::try_from(t).unwrap_or(-1))
            .collect();
        check_host_ids(&ids, *vocab)?;
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        ids_buf.write_at(stream, 0, &bytes)?;
        let dst = if streams == 1 { resid } else { scratch_rows };
        self.table.gather(
            ctx,
            stream,
            [
                ids_buf.range(0, n * 4)?,
                dst.range(0, n * hidden * 4)?,
                status_buf.range(0, 4)?,
            ],
            n,
        )?;
        if streams > 1 {
            ctx.block_op(
                stream,
                BlockOp::hc_broadcast(
                    n,
                    streams,
                    hidden,
                    scratch_rows.range(0, n * hidden * 4)?,
                    resid.range(0, n * streams * hidden * 4)?,
                )?,
            )?;
        }
        Ok(())
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

/// Whether `sampler` can draw on the device via the host-equal top-20 shortlist (for `top_k` in
/// `1..=20`) or the top-20 shortlist with softmax-mass certificate (for open-vocabulary `top_k == 0`
/// with a finite positive f32 inverse temperature).
pub(crate) fn can_shortlist_sample(sampler: &Sampler) -> bool {
    if sampler.capture_armed() || sampler.is_greedy() {
        return false;
    }
    let params = sampler.params();
    if (1..=shortlist::K as u32).contains(&params.top_k()) {
        return true;
    }
    if sampler.open_draw_on_device() {
        let inv_temp = (1.0 / params.temperature()) as f32;
        return inv_temp.is_finite() && inv_temp > 0.0;
    }
    false
}

/// Encodes f32 values as little-endian bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decodes little-endian bytes into f32 values.
pub(crate) fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// Reads the first `vocab` f32 logits of `logits` from the device on `stream`.
pub(crate) fn read_logits_row(
    stream: &DeviceStream,
    logits: &DeviceBuffer,
    vocab: usize,
) -> Result<Vec<f32>, EngError> {
    let byte_len = vocab
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| EngError::BadRequest("logits row byte length overflows".to_owned()))?;
    if byte_len == 0 || byte_len > logits.bytes() {
        return Err(EngError::BadRequest(format!(
            "logits read of {vocab} values exceeds buffer of {} bytes",
            logits.bytes()
        )));
    }
    let mut bytes = vec![0u8; byte_len];
    logits.read_at(stream, 0, &mut bytes)?;
    Ok(f32_values(&bytes))
}

fn decode_shortlist(bytes: &[u8], vocab: usize) -> Result<Vec<(u32, f32)>, EngError> {
    if bytes.len() < shortlist::RESULT_BYTES {
        return Err(EngError::Backend {
            operation: "sample",
            message: "shortlist readback buffer is truncated".to_owned(),
        });
    }
    if bytes[160..164] != [0; 4] {
        return Err(EngError::Backend {
            operation: "sample",
            message: "device shortlist found non-finite logits".to_owned(),
        });
    }
    Ok(bytes[..vocab.min(shortlist::K) * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|entry| {
            (
                u32::from_le_bytes(entry[4..8].try_into().expect("candidate id word")),
                f32::from_le_bytes(entry[..4].try_into().expect("candidate logit word")),
            )
        })
        .collect())
}

/// Draws one token from the logits row already on the device using the host-equal shortlist
/// (`top_k` 1..=20, 176-byte readback) or the shortlist plus softmax-mass certificate
/// (`top_k == 0`, 240-byte readback, reading the full logits row only when the certificate
/// declines). The token stream equals the host sampler's for every seed.
pub(crate) fn sample_shortlist(
    ctx: &CudaContext,
    stream: &DeviceStream,
    logits: &DeviceBuffer,
    scratch: &DeviceBuffer,
    sampler: &mut Sampler,
    sampler_vocab: usize,
) -> Result<u32, EngError> {
    let logits_bytes = sampler_vocab
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| EngError::BadRequest("device shortlist vocabulary overflows".to_owned()))?;
    if sampler_vocab == 0 || logits_bytes > logits.bytes() {
        return Err(EngError::BadRequest(format!(
            "device shortlist vocabulary {sampler_vocab} is outside 1..={}",
            logits.bytes() / size_of::<f32>()
        )));
    }
    let logits_range = logits.range(0, logits_bytes)?;
    ctx.shortlist(
        stream,
        ShortlistCall {
            k: shortlist::K,
            logits: logits_range,
            scratch: scratch.range(0, shortlist::SCRATCH_BYTES)?,
            n: sampler_vocab,
        },
    )?;
    if sampler.open_draw_on_device() {
        let inv_temperature = (1.0 / sampler.params().temperature()) as f32;
        ctx.shortlist_mass(
            stream,
            ShortlistMassCall {
                k: shortlist::K,
                logits: logits_range,
                scratch: scratch.range(0, shortlist::MASS_SCRATCH_BYTES)?,
                n: sampler_vocab,
                inv_temperature,
            },
        )?;
        // Final shortlist record (RESULT_BYTES) and MASS_PARTS f64 partials sit contiguously.
        const _: () =
            assert!(shortlist::RESULT_OFFSET + shortlist::RESULT_BYTES == shortlist::MASS_OFFSET);
        const COMBINED: usize = shortlist::RESULT_BYTES + shortlist::MASS_PARTS * 8;
        let mut buf = [0u8; COMBINED];
        scratch.read_at(stream, shortlist::RESULT_OFFSET, &mut buf)?;
        let candidates = decode_shortlist(&buf[..shortlist::RESULT_BYTES], sampler_vocab)?;
        let mass: f64 = buf[shortlist::RESULT_BYTES..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|part| f64::from_le_bytes(*part))
            .sum();
        return sampler.sample_open_shortlist(&candidates, sampler_vocab, mass, || {
            read_logits_row(stream, logits, sampler_vocab)
        });
    }
    let mut buf = [0u8; shortlist::RESULT_BYTES];
    scratch.read_at(stream, shortlist::RESULT_OFFSET, &mut buf)?;
    let candidates = decode_shortlist(&buf, sampler_vocab)?;
    sampler.sample_shortlist(&candidates, sampler_vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::sampler::SamplingParams;

    #[test]
    fn can_shortlist_sample_gates_bounded_open_and_special_samplers() {
        let greedy = Sampler::greedy();
        assert!(!can_shortlist_sample(&greedy));

        let k1 = Sampler::new(SamplingParams::new(0.8, 0.9, 1).unwrap(), 1);
        assert!(can_shortlist_sample(&k1));
        let k20 = Sampler::new(SamplingParams::new(0.8, 0.9, 20).unwrap(), 1);
        assert!(can_shortlist_sample(&k20));
        let k21 = Sampler::new(SamplingParams::new(0.8, 0.9, 21).unwrap(), 1);
        assert!(!can_shortlist_sample(&k21));

        let open = Sampler::new(SamplingParams::new(0.8, 0.95, 0).unwrap(), 1);
        assert!(can_shortlist_sample(&open));
        let open_p0 = Sampler::new(SamplingParams::new(0.8, 0.0, 0).unwrap(), 1);
        assert!(can_shortlist_sample(&open_p0));

        // Tiny temperature whose f32 reciprocal overflows to +inf cannot run shortlist_mass,
        // while bounded top_k <= 20 computes softmax in f64 on host and remains eligible.
        let open_tiny_t = Sampler::new(SamplingParams::new(1e-45, 0.95, 0).unwrap(), 1);
        assert!(!can_shortlist_sample(&open_tiny_t));
        let bounded_tiny_t = Sampler::new(SamplingParams::new(1e-45, 0.95, 10).unwrap(), 1);
        assert!(can_shortlist_sample(&bounded_tiny_t));

        let mut captured = Sampler::new(SamplingParams::new(0.8, 0.9, 10).unwrap(), 1);
        let _cap = captured.arm_capture(4);
        assert!(!can_shortlist_sample(&captured));
    }

    #[test]
    fn decode_shortlist_validates_bounds_finiteness_and_layout() {
        let short = [0u8; shortlist::RESULT_BYTES - 1];
        assert!(matches!(
            decode_shortlist(&short, 5),
            Err(EngError::Backend {
                operation: "sample",
                ..
            })
        ));

        let mut bad = [0u8; shortlist::RESULT_BYTES];
        bad[160..164].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            decode_shortlist(&bad, 5),
            Err(EngError::Backend {
                operation: "sample",
                ..
            })
        ));

        let mut ok = [0u8; shortlist::RESULT_BYTES];
        let entries = [(42u32, 3.25f32), (7, -1.5), (99, 0.0)];
        for (i, &(id, logit)) in entries.iter().enumerate() {
            ok[i * 8..i * 8 + 4].copy_from_slice(&logit.to_le_bytes());
            ok[i * 8 + 4..i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        assert_eq!(decode_shortlist(&ok, 2).unwrap(), entries[..2]);
        assert_eq!(decode_shortlist(&ok, 3).unwrap(), entries);
    }
}
