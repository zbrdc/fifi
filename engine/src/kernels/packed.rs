use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

use super::{QuantizationMode, QuantizationSpec};

#[derive(Debug)]
pub struct PackedWeight {
    payload: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    logical_shape: Vec<u32>,
    quantization: QuantizationSpec,
}

impl PackedWeight {
    pub fn try_new(
        payload: Tensor,
        scales: Tensor,
        biases: Option<Tensor>,
        logical_shape: Vec<u32>,
        quantization: QuantizationSpec,
    ) -> Result<Self, EngError> {
        quantization.validate()?;
        if logical_shape.len() < 2 || logical_shape.contains(&0) {
            return Err(EngError::Checkpoint(
                "logical shape requires rank at least two and positive dimensions".into(),
            ));
        }
        logical_shape.iter().try_fold(1u64, |count, &dimension| {
            count
                .checked_mul(u64::from(dimension))
                .ok_or_else(|| EngError::Checkpoint("logical element count overflows u64".into()))
        })?;
        let width = logical_shape[logical_shape.len() - 1];
        if width.checked_rem(quantization.group_size) != Some(0) {
            return Err(EngError::Checkpoint(
                "logical width must contain complete quantization groups".into(),
            ));
        }
        if payload.dtype() != Dtype::U32 {
            return Err(EngError::Checkpoint("payload dtype must be U32".into()));
        }
        let packed_bits = u64::from(width)
            .checked_mul(u64::from(quantization.bits))
            .ok_or_else(|| EngError::Checkpoint("packed width overflows u64".into()))?;
        if !packed_bits.is_multiple_of(32) {
            return Err(EngError::Checkpoint(
                "packed width requires complete U32 words".into(),
            ));
        }
        let packed_width = u32::try_from(packed_bits / 32)
            .map_err(|_| EngError::Checkpoint("packed width exceeds u32".into()))?;
        let mut expected_shape = logical_shape.clone();
        let last = expected_shape.len() - 1;
        expected_shape[last] = packed_width;
        if payload.shape() != expected_shape {
            return Err(EngError::Checkpoint(format!(
                "payload shape must be {expected_shape:?}, received {:?}",
                payload.shape()
            )));
        }
        if quantization.mode == QuantizationMode::Affine
            && !matches!(scales.dtype(), Dtype::F16 | Dtype::Bf16 | Dtype::F32)
        {
            return Err(EngError::Checkpoint(
                "affine scales dtype must be F16, Bf16, or F32".into(),
            ));
        }
        if quantization.mode != QuantizationMode::Affine && scales.dtype() != Dtype::U8 {
            return Err(EngError::Checkpoint("MXFP scales dtype must be U8".into()));
        }
        if quantization.mode == QuantizationMode::Affine {
            let biases = biases.as_ref().ok_or_else(|| {
                EngError::Checkpoint("affine weights require additive biases".into())
            })?;
            if biases.dtype() != scales.dtype() {
                return Err(EngError::Checkpoint(
                    "affine biases dtype must match scales".into(),
                ));
            }
        }
        if quantization.mode != QuantizationMode::Affine && biases.is_some() {
            return Err(EngError::Checkpoint(
                "MXFP weights must not contain additive biases".into(),
            ));
        }
        expected_shape[last] = width / quantization.group_size;
        for (role, tensor) in [("scales", Some(&scales)), ("biases", biases.as_ref())] {
            if let Some(tensor) = tensor
                && tensor.shape() != expected_shape
            {
                return Err(EngError::Checkpoint(format!(
                    "{role} shape must be {expected_shape:?}, received {:?}",
                    tensor.shape()
                )));
            }
        }
        Ok(Self {
            payload,
            scales,
            biases,
            logical_shape,
            quantization,
        })
    }

    pub fn quantization(&self) -> QuantizationSpec {
        self.quantization
    }
    pub fn logical_shape(&self) -> &[u32] {
        &self.logical_shape
    }
    pub fn payload(&self) -> &Tensor {
        &self.payload
    }
    pub fn scales(&self) -> &Tensor {
        &self.scales
    }
    pub fn biases(&self) -> Option<&Tensor> {
        self.biases.as_ref()
    }
}
