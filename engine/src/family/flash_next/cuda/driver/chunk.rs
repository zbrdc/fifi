//! The chunk driver: one prompt chunk of `m` tokens end to end, behind the same rules as the
//! token driver.
//!
//! WHAT: [`ChunkDriver`] mirrors [`super::TokenDriver`] at chunk granularity — the prefill
//! counterpart of the decode step. Per chunk:
//!
//! 1. [`super::TokenFeeds::begin`] per token (host reads; a batched pack read can replace the
//!    loop later without touching this contract).
//! 2. The chunk start: step parameters `(token0, pos, m)`, the `m` stream rows, the `m` PLE
//!    embeddings.
//! 3. The layers before the PLE layer, `m` rows at `pos`.
//! 4. ONE host roundtrip: the `m` stream rows come back, each row gets its own
//!    [`super::TokenFeeds::ple_update`] on the host, and the rows return in one write.
//! 5. The remaining layers.
//! 6. With logits wanted: the final mixer and `lm_head` over the LAST row only, and the read-back
//!    of its logits row.
//! 7. The health check and the position advance by `m`.
//!
//! The rules carry over from the token driver unchanged: the fed count advances only when the
//! model state consumed the whole chunk; a failure after the first write to the card poisons the
//! driver (every later chunk is refused until `reset`); a refused id or an over-long chunk
//! changes nothing and does not poison.
//!
//! The device half is [`ChunkDevice`]. It exists so the sequence, the poison rule and the
//! position accounting run in host tests against fakes; the card implementation is staged for the
//! chunked prefill step (`step`/`mixer` at `m` rows).

use std::ops::Range;

use super::{DriverShape, TokenFeeds};
use crate::{EngError, family::session::TokenStepper};

type Result<T> = std::result::Result<T, EngError>;

/// The device side of a chunk. The start writes are the sink of the host feeds at chunk
/// granularity; every buffer holds `m` rows where the token step holds one.
pub(in crate::family::flash_next::cuda) trait ChunkDevice {
    /// Writes the step parameters of the chunk: its first token id, its start position and its
    /// row count.
    fn write_chunk_params(&mut self, token0: i32, pos: i32, m: i32) -> Result<()>;
    /// Writes the `m` stream rows (`m * branches * hidden`), the initial stream of the chunk.
    fn write_chunk_stream(&mut self, rows: &[f32]) -> Result<()>;
    /// Writes the `m` PLE embedding rows (`m * ple_embed`).
    fn write_chunk_ple(&mut self, rows: &[f32]) -> Result<()>;
    /// Clears the per-chunk flags (the EXL3 flag and the status word). Called once per chunk,
    /// after the start is written and before the first layer.
    fn begin_chunk(&mut self) -> Result<()>;
    /// Re-plans the prefill expert ring against the bank as decode left it. Called once at the
    /// start of a chunked prefill (a `feed_rows` call), before the first chunk.
    fn begin_prefill(&mut self) -> Result<()> {
        Ok(())
    }
    /// Closes the prefill window after the sampling row ran. The single-token step of later
    /// decode uses the bank's miss path again; a no-op when no window is open.
    fn end_prefill(&mut self) -> Result<()> {
        Ok(())
    }
    /// Queues layers `range` for the chunk of `m` rows at `pos`.
    fn layers(&mut self, range: Range<usize>, pos: usize, m: usize) -> Result<()>;
    /// The `m` stream rows, read back behind the queued work.
    fn read_stream(&mut self) -> Result<Vec<f32>>;
    /// Replaces the `m` stream rows.
    fn write_stream_back(&mut self, rows: &[f32]) -> Result<()>;
    /// Queues the final mixer and the head over the last row of the chunk.
    fn head(&mut self) -> Result<()>;
    /// The logits row of the last row's head, behind the queued work.
    fn read_logits(&mut self) -> Result<Vec<f32>>;
    /// Reads the status word and the arithmetic flag; refuses a chunk that set either.
    fn check_health(&mut self) -> Result<()>;
    /// Zeroes the persistent state: a new sequence.
    fn reset(&mut self) -> Result<()>;
    /// The bank's miss count for the per-job report, if the device has a bank.
    fn bank_misses(&self) -> Option<u64> {
        None
    }
}

/// The chunk driver: a [`ChunkDevice`] and its [`TokenFeeds`] under the token driver's rules.
/// `feed_rows` splits the prompt into device chunks of at most `chunk_cap` rows (1 = the token
/// loop, the decode shape).
pub(in crate::family::flash_next::cuda) struct ChunkDriver<D: ChunkDevice, F: TokenFeeds> {
    device: D,
    feeds: F,
    shape: DriverShape,
    /// Tokens the state has consumed: the next position.
    pos: usize,
    poisoned: bool,
    /// The most rows one device chunk runs; the caller sets it from the memory plan.
    pub(in crate::family::flash_next::cuda) chunk_cap: usize,
}

impl<D: ChunkDevice, F: TokenFeeds> ChunkDriver<D, F> {
    pub(in crate::family::flash_next::cuda) fn new(
        device: D,
        feeds: F,
        shape: DriverShape,
    ) -> Result<Self> {
        if shape.layers <= shape.ple_layer {
            return Err(EngError::BadRequest(format!(
                "Flash-Next chunk driver: the PLE layer {} must sit inside the {} layers",
                shape.ple_layer, shape.layers
            )));
        }
        Ok(Self {
            device,
            feeds,
            shape,
            pos: 0,
            poisoned: false,
            chunk_cap: 1,
        })
    }

    /// The number of positions the caches hold, for the executor.
    pub(super) fn positions(&self) -> usize {
        self.shape.positions
    }

    pub(in crate::family::flash_next::cuda) fn device(&self) -> &D {
        &self.device
    }

    pub(super) fn feeds(&self) -> &F {
        &self.feeds
    }

    /// The chunk of `tokens` (at least one): feeds every token, runs the layers with one PLE
    /// roundtrip, and returns the logits of the LAST row when `want_logits`.
    pub(super) fn run_chunk(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let m = tokens.len();
        if self.poisoned {
            return Err(EngError::BadRequest(
                "Flash-Next chunk driver: the state needs a reset after a failed step".into(),
            ));
        }
        if m == 0 {
            return Err(EngError::BadRequest(
                "Flash-Next chunk driver: a chunk needs at least one token".into(),
            ));
        }
        if self.pos + m > self.shape.positions {
            return Err(EngError::ContextLimit(format!(
                "Flash-Next chunk driver: the caches hold {} positions, a chunk of {m} at {} does not fit",
                self.shape.positions, self.pos
            )));
        }
        // A refused id changes nothing; the chunk start repeats the rule, but inside the queued
        // part, where a failure poisons the driver.
        if tokens.iter().any(|token| {
            usize::try_from(*token).is_ok_and(|id| id >= self.shape.start.sampler_vocab)
        }) {
            return Err(EngError::BadRequest(format!(
                "Flash-Next chunk driver: a token id is outside the sampler vocabulary 0..{}",
                self.shape.start.sampler_vocab
            )));
        }
        // Host reads only: a failure here leaves the state as it was.
        let starts = self.feeds.begin_chunk(tokens)?;
        let outcome = self.queue(tokens, &starts, want_logits);
        match outcome {
            Ok(logits) => {
                self.pos += m;
                Ok(logits)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    /// Everything after the first write to the card.
    fn queue(
        &mut self,
        tokens: &[u32],
        starts: &[super::StartVectors],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let m = tokens.len();
        let pos = self.pos;
        let shape = self.shape;

        self.device
            .write_chunk_params(tokens[0] as i32, pos as i32, m as i32)?;
        let mut stream_rows = Vec::with_capacity(m * shape.start.stream);
        let mut ple_rows = Vec::with_capacity(m * shape.start.ple_embed);
        for start in starts {
            stream_rows.extend_from_slice(&start.stream);
            ple_rows.extend_from_slice(&start.ple_embedding);
        }
        self.device.write_chunk_stream(&stream_rows)?;
        self.device.write_chunk_ple(&ple_rows)?;
        self.device.begin_chunk()?;

        self.device.layers(0..shape.ple_layer, pos, m)?;
        let mut rows = self.device.read_stream()?;
        let width = shape.start.stream;
        if rows.len() != m * width {
            return Err(EngError::Backend {
                operation: "flash_next_cuda_chunk",
                message: format!(
                    "the chunk read back {} stream values for {m} rows of {width}",
                    rows.len()
                ),
            });
        }
        for (i, token) in tokens.iter().enumerate() {
            let update = self
                .feeds
                .ple_update(*token, &rows[i * width..(i + 1) * width])?;
            if update.len() != width {
                return Err(EngError::BadRequest(format!(
                    "Flash-Next chunk driver: the PLE update has {} values for a stream of {width}",
                    update.len()
                )));
            }
            for (value, delta) in rows[i * width..(i + 1) * width].iter_mut().zip(&update) {
                *value += delta;
            }
        }
        self.device.write_stream_back(&rows)?;
        self.device.layers(shape.ple_layer..shape.layers, pos, m)?;
        let logits = if want_logits {
            self.device.head()?;
            let row = self.device.read_logits()?;
            Some(row)
        } else {
            None
        };
        self.device.check_health()?;
        Ok(logits)
    }
}

impl<D: ChunkDevice, F: TokenFeeds> super::StepperPositions for ChunkDriver<D, F> {
    fn positions(&self) -> usize {
        self.positions()
    }
}

impl<D: ChunkDevice, F: TokenFeeds> TokenStepper for ChunkDriver<D, F> {
    const OPERATION: &'static str = "flash_next_cuda_chunk";

    fn feed(&mut self, token: u32) -> Result<()> {
        self.run_chunk(&[token], false).map(|_| ())
    }

    /// The prefill path: the whole prompt chunk lands as device chunks of at most `chunk_cap`
    /// rows, so the executor's `feed_rows` call is the chunked prefill entry.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let cap = self.chunk_cap.max(1);
        self.device.begin_prefill()?;
        for chunk in tokens.chunks(cap) {
            self.run_chunk(chunk, false).map(|_| ())?;
        }
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        let logits = self
            .run_chunk(&[token], true)?
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })?;
        // The sampling row closed the prefill: the ring's window ends with it, and every
        // later single-token step takes the decode path.
        self.device.end_prefill()?;
        Ok(logits)
    }

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<()> {
        self.device.reset()?;
        self.feeds.reset();
        self.pos = 0;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
