//! Host decoding of ModelOpt NVFP4 weights. This does not implement activation quantization.

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

pub const GROUP_SIZE: usize = 16;

// NVIDIA ModelOpt 0.37.0 NVFP4QTensor.quantize packs (odd << 4) | even:
// https://github.com/NVIDIA/Model-Optimizer/blob/0.37.0/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L225
// Real-shard confirmation is pending.
const NIBBLE_SHIFTS: [u32; 2] = [0, 4];

// E2M1: magnitudes for codes 0..7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 supplies the sign.
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E4M3FN values, including signed zero and NaN at 0x7f/0xff; this format has no infinity.
pub const E4M3_VALUES: [f32; 256] = e4m3_table();

const fn e4m3_table() -> [f32; 256] {
    let mut values = [0.0; 256];
    let mut code = 0u32;
    while code < 256 {
        let exponent = (code >> 3) & 15;
        let fraction = code & 7;
        // Bias 7: normal = (1 + fraction/8) * 2^(exponent-7); subnormal = fraction * 2^-9.
        let magnitude = if exponent == 0 {
            fraction as f32 / 512.0
        } else if exponent == 15 && fraction == 7 {
            f32::NAN
        } else {
            f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
        };
        values[code as usize] = if code & 128 == 0 {
            magnitude
        } else {
            -magnitude
        };
        code += 1;
    }
    values
}

/// Scale tensor suffixes accepted by this weight codec. FP8 KV-cache scales are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleRole {
    WeightScale,
    WeightScale2,
    InputScale,
}

impl ScaleRole {
    pub fn parse(name: &str) -> Result<Self, EngError> {
        match name {
            "weight_scale" => Ok(Self::WeightScale),
            "weight_scale_2" => Ok(Self::WeightScale2),
            "input_scale" => Ok(Self::InputScale),
            _ => Err(bad(format!("unknown scale role: {name}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::WeightScale => "weight_scale",
            Self::WeightScale2 => "weight_scale_2",
            Self::InputScale => "input_scale",
        }
    }
}

/// Row-major packed weights with one E4M3FN byte per consecutive group of 16 elements.
#[derive(Debug, Clone, Copy)]
pub struct WeightBlock<'a> {
    pub packed: &'a [u8],
    pub weight_scale: &'a [u8],
    /// F32 per-tensor weight multiplier, not its reciprocal.
    pub weight_scale_2: f32,
    /// F32 activation scale. Retained and validated, but never applied to decoded weights.
    pub input_scale: Option<f32>,
}

impl WeightBlock<'_> {
    /// Decodes to F32, or F16 when the existing native-mlx half support is enabled on Apple Silicon.
    /// Each row must contain complete groups. Output bytes use little-endian tensor storage.
    pub fn decode(&self, shape: &[u32], dtype: Dtype) -> Result<Tensor, EngError> {
        let width = match dtype {
            Dtype::F32 => 4,
            #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
            Dtype::F16 => 2,
            _ => return Err(bad("unsupported output dtype")),
        };
        let byte_len = self.output_byte_len(shape, width)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| bad("output allocation exceeds host limits"))?;
        for (group, &code) in self
            .packed
            .as_chunks::<{ GROUP_SIZE / 2 }>()
            .0
            .iter()
            .zip(self.weight_scale)
        {
            let group_scale = E4M3_VALUES[usize::from(code)];
            if !group_scale.is_finite() || group_scale.is_sign_negative() {
                return Err(bad("weight_scale must be finite and nonnegative"));
            }
            let scale = group_scale * self.weight_scale_2;
            if !scale.is_finite() {
                return Err(bad("weight_scale product is not finite"));
            }
            for &packed in group {
                for shift in NIBBLE_SHIFTS {
                    let value = E2M1_VALUES[usize::from((packed >> shift) & 15)] * scale;
                    if !value.is_finite() {
                        return Err(bad("decoded weight is not finite"));
                    }
                    match dtype {
                        Dtype::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                        #[cfg(all(
                            feature = "native-mlx",
                            target_os = "macos",
                            target_arch = "aarch64"
                        ))]
                        Dtype::F16 => {
                            let value = half::f16::from_f32(value);
                            if !value.is_finite() {
                                return Err(bad("decoded weight exceeds finite F16 range"));
                            }
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        _ => unreachable!("output dtype was checked"),
                    }
                }
            }
        }
        Tensor::try_from_host(dtype, shape.to_vec(), bytes)
    }

    /// Computes y[r] = sum over c of W[r, c] * x[c] without materializing the matrix.
    ///
    /// The row is walked in column order, so the sum matches a decoded row dotted
    /// with x in the same order. Each group scale is validated once.
    pub fn matvec(
        &self,
        rows: usize,
        columns: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), EngError> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(GROUP_SIZE) {
            return Err(bad("matrix dimensions must be positive and group aligned"));
        }
        if x.len() != columns || y.len() != rows {
            return Err(bad(
                "activation or output length differs from the matrix shape",
            ));
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(bad("activation values must be finite"));
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| bad("matrix element count overflows host limits"))?;
        let per_row = columns / GROUP_SIZE;
        let groups = rows
            .checked_mul(per_row)
            .ok_or_else(|| bad("group count overflows host limits"))?;
        if self.packed.len() != elements / 2 || self.weight_scale.len() != groups {
            return Err(bad("stored bytes differ from the matrix shape"));
        }
        if !self.weight_scale_2.is_finite() || self.weight_scale_2 <= 0.0 {
            return Err(bad("weight_scale_2 must be finite and positive"));
        }
        let group_bytes = GROUP_SIZE / 2;
        for (row, out) in y.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for group in 0..per_row {
                let index = row * per_row + group;
                let group_scale = E4M3_VALUES[usize::from(self.weight_scale[index])];
                if !group_scale.is_finite() || group_scale.is_sign_negative() {
                    return Err(bad("weight_scale must be finite and nonnegative"));
                }
                let scale = group_scale * self.weight_scale_2;
                if !scale.is_finite() {
                    return Err(bad("weight_scale product is not finite"));
                }
                let start = index * group_bytes;
                let base = group * GROUP_SIZE;
                for (offset, &byte) in self.packed[start..start + group_bytes].iter().enumerate() {
                    for shift in NIBBLE_SHIFTS {
                        let column = base + offset * 2 + usize::from(shift != 0);
                        let value = E2M1_VALUES[usize::from((byte >> shift) & 15)] * scale;
                        if !value.is_finite() {
                            return Err(bad("decoded weight is not finite"));
                        }
                        sum += value * x[column];
                    }
                }
            }
            *out = sum;
        }
        Ok(())
    }

    fn output_byte_len(&self, shape: &[u32], width: usize) -> Result<usize, EngError> {
        let columns = shape.last().copied().unwrap_or(0);
        if columns == 0 || !columns.is_multiple_of(GROUP_SIZE as u32) {
            return Err(bad("shape must end in a positive multiple of 16"));
        }
        let elements = shape.iter().try_fold(1usize, |count, &dim| {
            count
                .checked_mul(dim as usize)
                .ok_or_else(|| bad("shape product overflows host limits"))
        })?;
        let byte_len = elements
            .checked_mul(width)
            .ok_or_else(|| bad("shape byte length overflows host limits"))?;
        if self.packed.len() != elements / 2 {
            return Err(bad("packed length does not match shape"));
        }
        if self.weight_scale.len() != elements / GROUP_SIZE {
            return Err(bad("weight_scale length does not match shape groups"));
        }
        if !self.weight_scale_2.is_finite() || self.weight_scale_2 <= 0.0 {
            return Err(bad("weight_scale_2 must be finite and positive"));
        }
        if let Some(scale) = self.input_scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(bad("input_scale must be finite and positive"));
        }
        Ok(byte_len)
    }
}

fn bad(message: impl Into<String>) -> EngError {
    EngError::BadRequest(format!("NVFP4: {}", message.into()))
}
