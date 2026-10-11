//! Output assembly over incremental decode.
//!
//! Interface and flush rules: the tokenizer design.

use super::stop::{StopHit, StopMatcher};
use crate::EngError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeFinish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, PartialEq, Eq)]
pub struct OutputDelta {
    pub text: String,
    pub finish: Option<DecodeFinish>,
}

/// One committed publication of `Executor::decode`.
///
/// `text` is the incremental detokenized text of `tokens`. It still contains any
/// stop string. When `finish` is `Some`, `text` also carries the decoder tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeBatch {
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish: Option<DecodeFinish>,
}

pub struct OutputAssembler {
    matcher: StopMatcher,
    stops: Vec<String>,
    held: String,
    emitted: usize,
    finished: Option<DecodeFinish>,
}

impl OutputAssembler {
    pub fn new(stops: Vec<String>) -> Result<Self, EngError> {
        let matcher = StopMatcher::new(stops.clone())?;
        Ok(Self {
            matcher,
            stops,
            held: String::new(),
            emitted: 0,
            finished: None,
        })
    }

    /// Matches stop strings on one batch of decoded text.
    ///
    /// `decode` already checked EOS, the output limit, and the context limit.
    /// A stop match wins over `batch.finish`, except that `Cancelled` keeps its
    /// reason. When `batch.finish` is `Some`, the delta releases every held byte.
    /// A call after the end returns an empty delta with the final reason.
    pub fn push_batch(&mut self, batch: &DecodeBatch) -> Result<OutputDelta, EngError> {
        if let Some(done) = self.finished {
            return Ok(Self::empty(done));
        }
        let hit = self.matcher.push_str(&batch.text);
        self.held.push_str(&batch.text);
        if hit.is_some() {
            let (text, stop) = self.take_text(hit);
            let finish = match batch.finish {
                Some(DecodeFinish::Cancelled) => DecodeFinish::Cancelled,
                _ => stop.unwrap_or(DecodeFinish::Stop),
            };
            self.finished = Some(finish);
            return Ok(OutputDelta {
                text,
                finish: Some(finish),
            });
        }
        if let Some(finish) = batch.finish {
            self.finished = Some(finish);
            return Ok(OutputDelta {
                text: self.release_all(),
                finish: Some(finish),
            });
        }
        let (text, _) = self.take_text(None);
        Ok(OutputDelta { text, finish: None })
    }

    pub fn finish(&mut self, reason: DecodeFinish) -> Result<OutputDelta, EngError> {
        if let Some(done) = self.finished {
            return Ok(Self::empty(done));
        }
        self.finished = Some(reason);
        Ok(OutputDelta {
            text: self.release_all(),
            finish: Some(reason),
        })
    }

    fn release_all(&mut self) -> String {
        self.emitted += self.held.len();
        std::mem::take(&mut self.held)
    }

    fn empty(finish: DecodeFinish) -> OutputDelta {
        OutputDelta {
            text: String::new(),
            finish: Some(finish),
        }
    }

    fn take_text(&mut self, hit: Option<StopHit>) -> (String, Option<DecodeFinish>) {
        let Some(hit) = hit else {
            let keep = self.hold_back();
            let tail = self.held.split_off(keep);
            let text = std::mem::replace(&mut self.held, tail);
            self.emitted += text.len();
            return (text, None);
        };
        let stop_len = self.stops[hit.stop].len();
        let start = hit.end.saturating_sub(stop_len);
        let keep = boundary_floor(
            &self.held,
            start.saturating_sub(self.emitted).min(self.held.len()),
        );
        let text = self.held[..keep].to_string();
        self.emitted += keep;
        self.held.clear();
        (text, Some(DecodeFinish::Stop))
    }

    fn hold_back(&self) -> usize {
        let bytes = self.held.as_bytes();
        for start in 0..bytes.len() {
            if !self.held.is_char_boundary(start) {
                continue;
            }
            if self
                .stops
                .iter()
                .any(|stop| stop.as_bytes().starts_with(&bytes[start..]))
            {
                return start;
            }
        }
        bytes.len()
    }
}

// A byte-exact stop match can begin inside a multi-byte character. The emitted
// text keeps whole characters only, so the boundary moves down to the nearest
// character start.
fn boundary_floor(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}
