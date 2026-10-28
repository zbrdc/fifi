use crate::EngError;

#[derive(Debug, PartialEq, Eq)]
pub struct StopHit {
    pub stop: usize,
    pub end: usize,
}

#[derive(Debug)]
pub struct StopMatcher {
    stops: Vec<String>,
    tail: Vec<u8>,
    stream_len: usize,
    done: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Result<Self, EngError> {
        if stops.iter().any(String::is_empty) {
            return Err(EngError::BadRequest(
                "stop string must not be empty".to_string(),
            ));
        }
        Ok(Self {
            stops,
            tail: Vec::new(),
            stream_len: 0,
            done: false,
        })
    }

    pub fn push_str(&mut self, chunk: &str) -> Option<StopHit> {
        if self.done {
            return None;
        }
        self.tail.extend_from_slice(chunk.as_bytes());
        self.stream_len += chunk.len();
        let hit = self.find_hit();
        if hit.is_some() {
            self.done = true;
            return hit;
        }
        self.trim_tail();
        None
    }

    fn find_hit(&self) -> Option<StopHit> {
        let base = self.stream_len - self.tail.len();
        let mut best: Option<StopHit> = None;
        for (index, text) in self.stops.iter().enumerate() {
            let bytes = text.as_bytes();
            let Some(start) = self.tail.windows(bytes.len()).position(|w| w == bytes) else {
                continue;
            };
            let hit = StopHit {
                stop: index,
                end: base + start + bytes.len(),
            };
            if best
                .as_ref()
                .is_none_or(|b| (hit.end, hit.stop) < (b.end, b.stop))
            {
                best = Some(hit);
            }
        }
        best
    }

    fn trim_tail(&mut self) {
        let keep = self
            .stops
            .iter()
            .map(|text| text.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        let drop = self.tail.len().saturating_sub(keep);
        self.tail.drain(..drop);
    }
}
