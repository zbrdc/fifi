//! Scalar, weight-free oracles and absolute-position window planning for Flint.
//!
//! The publisher's pinned `flint_sglang-0.1.4` model defines `softmax_offset`
//! as a per-query-head *logit*, independent of the QK scale. Its FA4 sink fix
//! rescales both the real-value accumulator and denominator when that logit
//! exceeds the real-key maximum. A sink is not a token and has no V vector.

use crate::EngError;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Unnormalized attention statistics for ONE query head, with no sink yet.
/// Merge any number of KV partitions, then call `finish` exactly once.
#[derive(Clone, Debug)]
pub struct AttentionPartial {
    max: f64,
    sum: f64,
    weighted: Vec<f64>,
}

impl AttentionPartial {
    pub fn new(value_dim: usize) -> Result<Self, EngError> {
        if value_dim == 0 {
            return Err(bad("attention value dimension must be positive"));
        }
        Ok(Self {
            max: f64::NEG_INFINITY,
            sum: 0.0,
            weighted: vec![0.0; value_dim],
        })
    }

    /// `score` is an already-scaled QK logit. Omit masked keys entirely.
    pub fn push(&mut self, score: f64, value: &[f64]) -> Result<(), EngError> {
        if value.len() != self.weighted.len()
            || !score.is_finite()
            || value.iter().any(|v| !v.is_finite())
        {
            return Err(bad(
                "attention score/value must be finite and match the value dimension",
            ));
        }
        let max = self.max.max(score);
        let old = (self.max - max).exp();
        let new = (score - max).exp();
        self.sum = self.sum * old + new;
        for (out, &v) in self.weighted.iter_mut().zip(value) {
            *out = *out * old + new * v;
        }
        self.max = max;
        Ok(())
    }

    pub fn merge(&mut self, other: &Self) -> Result<(), EngError> {
        if self.weighted.len() != other.weighted.len() {
            return Err(bad("attention partition value dimensions differ"));
        }
        if other.sum == 0.0 {
            return Ok(());
        }
        if self.sum == 0.0 {
            *self = other.clone();
            return Ok(());
        }
        let max = self.max.max(other.max);
        let a = (self.max - max).exp();
        let b = (other.max - max).exp();
        self.sum = self.sum * a + other.sum * b;
        for (out, &v) in self.weighted.iter_mut().zip(&other.weighted) {
            *out = *out * a + v * b;
        }
        self.max = max;
        Ok(())
    }

    /// Adds the sink once, after all KV partitions have been merged. An empty
    /// real-key set returns zero, including when all keys were masked.
    pub fn finish(self, sink_logit: f64) -> Result<Vec<f64>, EngError> {
        if !sink_logit.is_finite() {
            return Err(bad("attention sink logit must be finite"));
        }
        let max = self.max.max(sink_logit);
        let rescale = (self.max - max).exp();
        let denominator = self.sum * rescale + (sink_logit - max).exp();
        let output: Vec<_> = self
            .weighted
            .into_iter()
            .map(|v| v * rescale / denominator)
            .collect();
        if output.iter().any(|x| !x.is_finite()) {
            return Err(bad("attention value accumulation overflowed"));
        }
        Ok(output)
    }
}

/// Inclusive-current-token local attention over a whole appended query chunk.
/// Uses absolute positions; physical ring slots never determine causality.
#[derive(Clone, Copy, Debug)]
pub struct ChunkWindow {
    window: usize,
    start: usize,
    end: usize,
}

impl ChunkWindow {
    pub fn new(window: usize, start: usize, rows: usize) -> Result<Self, EngError> {
        if window == 0 || rows == 0 {
            return Err(bad("window and chunk rows must be positive"));
        }
        let end = start
            .checked_add(rows)
            .ok_or_else(|| bad("chunk position overflow"))?;
        Ok(Self { window, start, end })
    }

    /// Union of all query windows; must survive until the earliest query runs.
    pub fn live_range(&self) -> std::ops::Range<usize> {
        self.start.saturating_sub(self.window - 1)..self.end
    }

    /// May be `window + chunk_rows - 1`, not merely `window`.
    pub fn required_slots(&self) -> usize {
        self.live_range().len()
    }

    pub fn query_range(&self, query: usize) -> Result<std::ops::Range<usize>, EngError> {
        if !(self.start..self.end).contains(&query) {
            return Err(bad("query lies outside chunk"));
        }
        Ok(query.saturating_sub(self.window - 1)..query + 1)
    }

    /// Modulo, not a power-of-two mask. Refuse a whole-chunk append that would
    /// overwrite rows still needed by its first query.
    pub fn slot(&self, position: usize, capacity: usize) -> Result<usize, EngError> {
        if capacity < self.required_slots() || !self.live_range().contains(&position) {
            return Err(bad(
                "ring capacity/position does not cover the live chunk union",
            ));
        }
        Ok(position % capacity)
    }
}

/// Elementwise SwiGLU oracle for separately projected gate and up rows.
/// The same activation serves routed, dense, and *ungated* shared experts.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, EngError> {
    if gate.len() != up.len() || gate.is_empty() || gate.iter().chain(up).any(|x| !x.is_finite()) {
        return Err(bad(
            "SwiGLU projections must be finite, nonempty, equal-width rows",
        ));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&g, &u)| crate::family::flash_next::math::silu(g) * u)
        .collect())
}

/// One D128 Q/K head, direct gamma RMSNorm followed by optional half-split
/// RoPE. Full layers pass None; local layers pass the absolute token position.
pub fn qk_head(
    values: &[f32],
    gamma: &[f32],
    eps: f32,
    position: Option<u32>,
) -> Result<Vec<f32>, EngError> {
    if values.len() != 128
        || gamma.len() != 128
        || !eps.is_finite()
        || eps <= 0.0
        || values.iter().chain(gamma).any(|v| !v.is_finite())
    {
        return Err(bad(
            "Flint QK head/gamma must be finite D128 with positive epsilon",
        ));
    }
    let mean = crate::family::flash_next::math::dot(values, values)? / 128.0;
    if !mean.is_finite() {
        return Err(bad("Flint QK norm overflowed"));
    }
    let divisor = (mean + eps).sqrt();
    let mut output: Vec<_> = values
        .iter()
        .zip(gamma)
        .map(|(&v, &g)| v / divisor * g)
        .collect();
    if let Some(pos) = position {
        crate::family::flash_next::math::rope_half(&mut output, 128, pos, 10_000.0)?;
    }
    Ok(output)
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
