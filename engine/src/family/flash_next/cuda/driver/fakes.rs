//! Fakes of the two collaborators of the driver, for host tests of the sequence.
//!
//! The device keeps a stream of four floats; layer `l` at position `p` adds `l + 1 + p` to every
//! value; the logits row has [`LOGITS`] values and value `i` is `stream[0] * i`, so the greedy pick
//! of a positive stream is the last id of the row, which lies above the sampler vocabulary.

use std::ops::Range;

use super::super::driver::StateSlot;
use super::{DriverShape, StartVectors, TokenDevice, TokenFeeds, chunk::ChunkDevice};
use crate::family::flash_next::cuda::ple_feed::PleKeyValue;
use crate::{
    EngError,
    family::flash_next::cuda::feeds::{StartShape, StartSink},
};

pub(in crate::family::flash_next::cuda) const LAYERS: usize = 4;
pub(in crate::family::flash_next::cuda) const PLE_LAYER: usize = 2;
pub(in crate::family::flash_next::cuda) const POSITIONS: usize = 6;
pub(in crate::family::flash_next::cuda) const STREAM: usize = 4;
pub(in crate::family::flash_next::cuda) const PLE_EMBED: usize = 3;
/// Width of the logits row, the head.
pub(in crate::family::flash_next::cuda) const LOGITS: usize = 8;
/// Ids the sampler may draw: the tokenizer's vocabulary.
pub(in crate::family::flash_next::cuda) const SAMPLER_VOCAB: usize = 6;

pub(in crate::family::flash_next::cuda) fn shape() -> DriverShape {
    DriverShape {
        layers: LAYERS,
        ple_layer: PLE_LAYER,
        positions: POSITIONS,
        start: StartShape {
            sampler_vocab: SAMPLER_VOCAB,
            stream: STREAM,
            ple_embed: PLE_EMBED,
        },
    }
}

#[derive(Default)]
pub(in crate::family::flash_next::cuda) struct FakeDevice {
    /// Every call, in order.
    pub(in crate::family::flash_next::cuda) log: Vec<String>,
    pub(in crate::family::flash_next::cuda) stream: Vec<f32>,
    /// The fixed-state image a resume reads and installs (one blob; the fake has no slots).
    pub(in crate::family::flash_next::cuda) fixed: Vec<u8>,
    /// The call that fails (its log name), once.
    pub(in crate::family::flash_next::cuda) fail: Option<&'static str>,
    pub(in crate::family::flash_next::cuda) resets: usize,
}

impl FakeDevice {
    fn step(&mut self, name: &'static str, detail: String) -> Result<(), EngError> {
        self.log
            .push(format!("{name} {detail}").trim_end().to_string());
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::Backend {
                operation: "fake_device",
                message: format!("{name} failed"),
            });
        }
        Ok(())
    }
}

impl StartSink for FakeDevice {
    fn write_step_params(&mut self, token: i32, pos: i32) -> Result<(), EngError> {
        self.step("params", format!("{token} {pos}"))
    }

    fn write_stream(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.stream = stream.to_vec();
        self.step("stream", String::new())
    }

    fn write_ple_embedding(&mut self, _embedding: &[f32]) -> Result<(), EngError> {
        self.step("ple_embedding", String::new())
    }
}

impl TokenDevice for FakeDevice {
    fn begin_token(&mut self) -> Result<(), EngError> {
        self.step("begin", String::new())
    }

    fn layers(&mut self, range: Range<usize>, pos: usize) -> Result<(), EngError> {
        for layer in range.clone() {
            for value in &mut self.stream {
                *value += (layer + 1 + pos) as f32;
            }
        }
        self.step("layers", format!("{}..{} @{pos}", range.start, range.end))
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_stream", String::new())?;
        Ok(self.stream.clone())
    }

    fn write_stream_back(&mut self, stream: &[f32]) -> Result<(), EngError> {
        self.stream = stream.to_vec();
        self.step("write_back", String::new())
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step("head", String::new())
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_logits", String::new())?;
        Ok((0..LOGITS).map(|i| self.stream[0] * i as f32).collect())
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step("health", String::new())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.resets += 1;
        self.stream.clear();
        self.step("reset", String::new())
    }
}

impl super::StateIo for FakeDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        Ok(vec![(StateSlot::PleRing, self.fixed.clone())])
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        let Some((_, bytes)) = parts.first() else {
            return Ok(());
        };
        self.fixed = bytes.clone();
        self.log.push("state_write".to_string());
        Ok(())
    }
}

/// The chunk fake: `m` rows of the stream; layer `l` at position `p` adds `l + 1 + p + row` to
/// every value of row `row`, so a row carries its own index and the PLE roundtrip is checkable
/// per row. The logits row follows the LAST row, as the head does.
#[derive(Default)]
pub(in crate::family::flash_next::cuda) struct FakeChunkDevice {
    /// Every call, in order.
    pub(in crate::family::flash_next::cuda) log: Vec<String>,
    pub(in crate::family::flash_next::cuda) stream: Vec<f32>,
    /// The device PLE pairs `ple_project_chunk` returns, one per row.
    pub(in crate::family::flash_next::cuda) ple_pairs: Option<Vec<PleKeyValue>>,
    /// The call that fails (its log name), once.
    pub(in crate::family::flash_next::cuda) fail: Option<&'static str>,
    pub(in crate::family::flash_next::cuda) resets: usize,
    /// The last state the resume seam wrote back, as read.
    pub(in crate::family::flash_next::cuda) state: Vec<(StateSlot, Vec<u8>)>,
}

impl FakeChunkDevice {
    fn step(&mut self, name: &'static str, detail: String) -> Result<(), EngError> {
        self.log
            .push(format!("{name} {detail}").trim_end().to_string());
        if self.fail == Some(name) {
            self.fail = None;
            return Err(EngError::Backend {
                operation: "fake_chunk_device",
                message: format!("{name} failed"),
            });
        }
        Ok(())
    }
}

impl super::StateIo for FakeChunkDevice {
    fn read_fixed_state(&self) -> Result<Vec<(StateSlot, Vec<u8>)>, EngError> {
        // read takes &self, so no log step here — the log is only for the mutable calls.
        Ok(self.state.clone())
    }

    fn write_fixed_state(&mut self, parts: &[(StateSlot, Vec<u8>)]) -> Result<(), EngError> {
        self.step("write_fixed_state", String::new())?;
        self.state = parts.to_vec();
        Ok(())
    }
}

impl ChunkDevice for FakeChunkDevice {
    fn write_chunk_params(&mut self, token0: i32, pos: i32, m: i32) -> Result<(), EngError> {
        self.step("chunk_params", format!("{token0} {pos} {m}"))
    }

    fn write_chunk_stream(&mut self, rows: &[f32]) -> Result<(), EngError> {
        self.stream = rows.to_vec();
        self.step("chunk_stream", String::new())
    }

    fn write_chunk_ple(&mut self, _rows: &[f32]) -> Result<(), EngError> {
        self.step("chunk_ple", String::new())
    }

    fn ple_project_chunk(&mut self) -> Result<Vec<PleKeyValue>, EngError> {
        self.step("chunk_ple_project", String::new())?;
        self.ple_pairs
            .clone()
            .ok_or_else(|| EngError::BadRequest("the fake has no PLE pairs".to_string()))
    }

    fn begin_chunk(&mut self) -> Result<(), EngError> {
        self.step("begin", String::new())
    }

    fn layers(&mut self, range: Range<usize>, pos: usize, m: usize) -> Result<(), EngError> {
        let width = self.stream.len() / m.max(1);
        for layer in range.clone() {
            for (row, value) in self.stream.chunks_mut(width).enumerate() {
                let add = (layer + 1 + pos + row) as f32;
                for cell in value.iter_mut() {
                    *cell += add;
                }
            }
        }
        self.step(
            "layers",
            format!("{}..{} @{pos} m{m}", range.start, range.end),
        )
    }

    fn read_stream(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_stream", String::new())?;
        Ok(self.stream.clone())
    }

    fn write_stream_back(&mut self, rows: &[f32]) -> Result<(), EngError> {
        self.stream = rows.to_vec();
        self.step("write_back", String::new())
    }

    fn head(&mut self) -> Result<(), EngError> {
        self.step("head", String::new())
    }

    fn read_logits(&mut self) -> Result<Vec<f32>, EngError> {
        self.step("read_logits", String::new())?;
        let width = self.stream.len().max(1);
        let last = self.stream[width - STREAM..].to_vec();
        Ok((0..LOGITS).map(|i| last[0] * i as f32).collect())
    }

    fn check_health(&mut self) -> Result<(), EngError> {
        self.step("health", String::new())
    }

    /// Logs like every other call, so the boundary-order test can see the drain between the
    /// phases; the `fail` mechanism covers a failing drain by name.
    fn drain(&mut self) -> Result<(), EngError> {
        self.step("drain", String::new())
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.resets += 1;
        self.stream.clear();
        self.step("reset", String::new())
    }
}

#[derive(Default)]
pub(in crate::family::flash_next::cuda) struct FakeFeeds {
    /// The updates asked for: the token and the stream it saw.
    pub(in crate::family::flash_next::cuda) updates: Vec<(u32, Vec<f32>)>,
    /// The first key of every device pair set (the device route hands them over in order).
    pub(in crate::family::flash_next::cuda) pairs: Vec<f32>,
    pub(in crate::family::flash_next::cuda) resets: usize,
    /// The token whose `begin` fails.
    pub(in crate::family::flash_next::cuda) fail_begin: Option<u32>,
    /// Fails the PLE update.
    pub(in crate::family::flash_next::cuda) fail_update: bool,
    /// The width of the update (the stream's, unless a test breaks it).
    pub(in crate::family::flash_next::cuda) update_width: Option<usize>,
}

impl TokenFeeds for FakeFeeds {
    fn begin(&mut self, token: u32) -> Result<StartVectors, EngError> {
        if self.fail_begin == Some(token) {
            return Err(EngError::Checkpoint("the pack read failed".into()));
        }
        Ok(StartVectors {
            stream: vec![token as f32; STREAM],
            ple_embedding: vec![token as f32; PLE_EMBED],
        })
    }

    fn ple_update(&mut self, token: u32, stream: &[f32]) -> Result<Vec<f32>, EngError> {
        if self.fail_update {
            return Err(EngError::Checkpoint("the PLE step failed".into()));
        }
        self.updates.push((token, stream.to_vec()));
        Ok(vec![0.5; self.update_width.unwrap_or(stream.len())])
    }

    fn set_ple_pair(&mut self, pair: PleKeyValue) -> Result<(), EngError> {
        self.pairs
            .push(pair.keys.first().copied().unwrap_or(f32::NAN));
        Ok(())
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}
