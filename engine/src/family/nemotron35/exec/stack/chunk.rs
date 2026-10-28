//! Opt-in layer-major prefill. Each chunk publishes state and its last prediction
//! together; a failed chunk leaves the preceding committed chunk untouched.

use std::ffi::OsStr;

use super::*;

pub(super) fn parse_rows(value: Option<&OsStr>) -> Result<Option<usize>, EngError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| invalid("NEM35_CUDA_CHUNK_PREFILL must be a nonnegative row count"))?;
    let rows = value
        .parse::<usize>()
        .map_err(|_| invalid("NEM35_CUDA_CHUNK_PREFILL row count overflow"))?;
    Ok((rows != 0).then_some(rows))
}

impl<S: WeightSource> NemotronStack<S> {
    pub(super) fn prefill_chunked(&mut self, tokens: &[u32], rows: usize) -> Result<(), EngError> {
        if rows == 0 {
            return Err(invalid("chunk rows must be positive"));
        }
        // Validate the entire request before even the first chunk can publish.
        if tokens.len() > (self.state().capacity - self.state().position) as usize {
            return Err(invalid("context capacity exceeded"));
        }
        let g = self.spec.geometry;
        if tokens.iter().any(|&token| token >= g.vocab_size) {
            return Err(invalid("token exceeds vocabulary"));
        }
        for tokens in tokens.chunks(rows) {
            let embedding = self.dense_weight(
                &format!("{ROOT}embeddings.weight"),
                &[g.vocab_size, g.hidden_size],
            )?;
            // Slice only the requested rows: a full-table read would
            // re-introduce the ~2 GiB finiteness scan per chunk.
            let width = g.hidden_size as usize;
            let table = embedding
                .host_bytes()
                .ok_or_else(|| invalid("embedding not host-backed"))?;
            let mut values = Vec::with_capacity(tokens.len() * width);
            for &token in tokens {
                let start = token as usize * width;
                let row: Vec<f32> = table[start * 4..(start + width) * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                values.extend(row);
            }
            // Request validation bounds the chunk length by the u32 capacity.
            let count = tokens.len() as u32;
            let hidden = tensor(&[count, g.hidden_size], &values)?;
            let mut next_state = self.state().clone();
            let hidden = self.run_blocks(hidden, &mut next_state)?;
            let values = read(&hidden, &[count, g.hidden_size])?;
            let last = tensor(&[1, g.hidden_size], &values[values.len() - width..])?;
            // Every committed chunk leaves a usable prediction (the
            // failure-atomicity contract); with large rows the intermediate
            // heads are a ~1% cost.
            self.publish_last_row(last, next_state, count, true)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
