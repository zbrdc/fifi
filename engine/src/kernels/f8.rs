//! Host decoding of per-tensor-scaled E4M3FN weights.

use crate::{
    EngError,
    kernels::nvfp4::E4M3_VALUES,
    tensor::{Dtype, Tensor},
};

/// E4M3FN weight bytes with one F32 weight scale and an optional retained input scale.
#[derive(Debug, Clone, Copy)]
pub struct WeightBlock<'a> {
    pub codes: &'a [u8],
    /// Little-endian storage for one positive F32 weight multiplier.
    pub weight_scale: &'a [u8],
    /// Optional little-endian storage for one positive F32 activation scale.
    pub input_scale: Option<&'a [u8]>,
}

impl WeightBlock<'_> {
    /// Decodes E4M3FN weights to F32 and returns the parsed input scale unchanged.
    pub fn decode(&self, shape: &[u32]) -> Result<(Tensor, Option<f32>), EngError> {
        let weight_scale = parse_scale(self.weight_scale, "weight_scale")?;
        let input_scale = self
            .input_scale
            .map(|bytes| parse_scale(bytes, "input_scale"))
            .transpose()?;
        let elements = shape.iter().try_fold(1usize, |count, &dimension| {
            let dimension = usize::try_from(dimension)
                .map_err(|_| bad("shape dimension exceeds host limits"))?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| bad("shape product overflows host limits"))
        })?;
        let byte_len = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| bad("shape byte length overflows host limits"))?;
        if self.codes.len() != elements {
            return Err(bad("codes length does not match shape"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| bad("output allocation exceeds host limits"))?;
        for &code in self.codes {
            let decoded = E4M3_VALUES[usize::from(code)];
            if !decoded.is_finite() {
                return Err(bad("weight code is NaN"));
            }
            let value = decoded * weight_scale;
            if !value.is_finite() {
                return Err(bad("decoded weight is not finite"));
            }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensor = Tensor::try_from_host(Dtype::F32, shape.to_vec(), bytes)?;
        Ok((tensor, input_scale))
    }
}

fn parse_scale(bytes: &[u8], name: &str) -> Result<f32, EngError> {
    let [low, next, third, high] = bytes else {
        return Err(bad(format!("{name} must contain one F32")));
    };
    let scale = f32::from_le_bytes([*low, *next, *third, *high]);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(bad(format!("{name} must be finite and positive")));
    }
    Ok(scale)
}

fn bad(message: impl Into<String>) -> EngError {
    EngError::BadRequest(format!("FP8: {}", message.into()))
}
