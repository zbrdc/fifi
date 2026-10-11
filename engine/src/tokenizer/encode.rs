use crate::EngError;
use std::sync::Arc;

pub struct Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    vocab_size: u32,
}

impl Tokenizer {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, EngError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(|error| {
            EngError::Checkpoint(format!("tokenizer.json parse failed: {error}"))
        })?;
        // `get_vocab_size(true)` builds the whole vocabulary map on each call.
        // The size is fixed after parsing, so the wrapper queries it once.
        let vocab_size = u32::try_from(inner.get_vocab_size(true)).map_err(|_| {
            EngError::Checkpoint("tokenizer vocabulary size does not fit in u32".to_string())
        })?;
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| EngError::Checkpoint(format!("prompt encode failed: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, EngError> {
        if ids.iter().any(|&id| id >= self.vocab_size) {
            return Err(EngError::Checkpoint(
                "token id outside tokenizer vocab".to_string(),
            ));
        }
        self.inner
            .decode(ids, true)
            .map_err(|error| EngError::Checkpoint(format!("token decode failed: {error}")))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub(super) fn shared(&self) -> Arc<tokenizers::Tokenizer> {
        Arc::clone(&self.inner)
    }

    #[cfg(test)]
    pub(super) fn raw(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
