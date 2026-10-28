//! Flint's pinned BF16 embedding policy; no global precision setting.

use crate::{
    EngError,
    kernels::{PackedWeight, QuantizationMode},
    tensor::{Dtype, Tensor},
};

/// Expand only the requested INT8 rows with the publisher GPU's single BF16
/// rounding after F32 multiply-add. Its CPU dequantizer rounds twice instead.
pub(super) fn embedding(packed: &PackedWeight) -> Result<Tensor, EngError> {
    let &[rows, 2048] = packed.logical_shape() else {
        return Err(EngError::Checkpoint(
            "Flint embedding rows must have width2048".into(),
        ));
    };
    let quant = packed.quantization();
    let bias = packed
        .biases()
        .ok_or_else(|| EngError::Checkpoint("Flint embedding bias is absent".into()))?;
    if rows == 0
        || rows > 8192
        || quant.bits != 8
        || quant.group_size != 64
        || quant.mode != QuantizationMode::Affine
        || packed.payload().dtype() != Dtype::U32
        || packed.scales().dtype() != Dtype::Bf16
        || bias.dtype() != Dtype::Bf16
    {
        return Err(EngError::Checkpoint(
            "Flint embedding requires bounded affine8/group64 BF16 parts".into(),
        ));
    }
    let host = |tensor: &Tensor| {
        tensor
            .host_bytes()
            .map(|b| b.to_vec())
            .ok_or_else(|| EngError::Checkpoint("Flint embedding requires host rows".into()))
    };
    let codes = host(packed.payload())?;
    let scales = host(packed.scales())?;
    let biases = host(bias)?;
    let count = (rows as usize)
        .checked_mul(2048)
        .ok_or_else(|| EngError::MemoryPlan("Flint embedding count overflow".into()))?;
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| EngError::MemoryPlan("Flint embedding byte overflow".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| EngError::MemoryPlan("Flint embedding row allocation failed".into()))?;
    let value = |data: &[u8], group: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([data[group * 2], data[group * 2 + 1]])) << 16)
    };
    for (index, &code) in codes.iter().enumerate() {
        let sum = f32::from(code).mul_add(value(&scales, index / 64), value(&biases, index / 64));
        let bits = bf16_bits(sum);
        if !f32::from_bits(u32::from(bits) << 16).is_finite() {
            return Err(EngError::Checkpoint(
                "Flint embedding dequantization is non-finite".into(),
            ));
        }
        output.extend(bits.to_le_bytes());
    }
    Tensor::try_from_host(Dtype::Bf16, vec![rows, 2048], output)
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use crate::{
        kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
        tensor::{Dtype, Tensor},
    };
    #[test]
    fn publisher_cpu_fixture_is_not_the_gpu_rounding_policy() {
        let packed = PackedWeight::try_new(
            Tensor::try_from_host(
                Dtype::U32,
                vec![2, 512],
                include_bytes!("../../../testdata/flint/precision/embedding-weight.bin").to_vec(),
            )
            .unwrap(),
            Tensor::try_from_host(
                Dtype::Bf16,
                vec![2, 32],
                include_bytes!("../../../testdata/flint/precision/embedding-scales.bin").to_vec(),
            )
            .unwrap(),
            Some(
                Tensor::try_from_host(
                    Dtype::Bf16,
                    vec![2, 32],
                    include_bytes!("../../../testdata/flint/precision/embedding-biases.bin")
                        .to_vec(),
                )
                .unwrap(),
            ),
            vec![2, 2048],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 8,
                group_size: 64,
            },
        )
        .unwrap();
        let actual = super::embedding(&packed).unwrap();
        assert_eq!(actual.dtype(), Dtype::Bf16);
        let golden =
            include_bytes!("../../../testdata/flint/precision/embedding-publisher-dequant.f32.bin");
        assert_eq!(golden.len(), 4096 * 4);
        let differs = actual
            .host_bytes()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .zip(golden.as_chunks::<4>().0)
            .any(|(bytes, expected)| {
                u32::from(u16::from_le_bytes(*bytes)) << 16 != u32::from_le_bytes(*expected)
            });
        assert!(differs);
    }
    #[test]
    fn embedding_matches_publisher_gpu_values_exactly() {
        let packed = PackedWeight::try_new(
            Tensor::try_from_host(
                Dtype::U32,
                vec![2, 512],
                include_bytes!("../../../testdata/flint/precision/embedding-gpu-weight.bin")
                    .to_vec(),
            )
            .unwrap(),
            Tensor::try_from_host(
                Dtype::Bf16,
                vec![2, 32],
                include_bytes!("../../../testdata/flint/precision/embedding-gpu-scales.bin")
                    .to_vec(),
            )
            .unwrap(),
            Some(
                Tensor::try_from_host(
                    Dtype::Bf16,
                    vec![2, 32],
                    include_bytes!("../../../testdata/flint/precision/embedding-gpu-biases.bin")
                        .to_vec(),
                )
                .unwrap(),
            ),
            vec![2, 2048],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 8,
                group_size: 64,
            },
        )
        .unwrap();
        let actual = super::embedding(&packed).unwrap();
        assert_eq!(actual.dtype(), Dtype::Bf16);
        let golden =
            include_bytes!("../../../testdata/flint/precision/embedding-gpu-publisher.f32.bin");
        assert_eq!(golden.len(), 4096 * 4);
        assert_eq!(actual.host_bytes().unwrap().len(), 4096 * 2);
        for (bytes, expected) in actual
            .host_bytes()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .zip(golden.as_chunks::<4>().0)
        {
            assert_eq!(
                u32::from(u16::from_le_bytes(*bytes)) << 16,
                u32::from_le_bytes(*expected)
            );
        }
    }
}
