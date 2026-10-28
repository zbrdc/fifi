//! Typed EXL3 upload seam; family routing remains off until model qualification.

use half::f16;

use super::{CudaProjector, Reserve};
#[cfg(test)]
use crate::weights::exl3::Checkpoint;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange, DeviceStream, resident::exl3::Exl3LinearCall,
    },
    tensor::{Dtype, Tensor},
    weights::exl3::{Codebook, Matrix},
};

pub(crate) struct Exl3Weight {
    // Reverse upload order also applies when the complete weight is dropped.
    pub(crate) bias: Option<DeviceBuffer>,
    pub(crate) svh: DeviceBuffer,
    pub(crate) suh: DeviceBuffer,
    pub(crate) trellis: DeviceBuffer,
    pub(crate) shape: [usize; 2],
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}

/// Borrowed packed components; the owner outlives every in-flight projection.
#[derive(Clone, Copy)]
pub(crate) struct Exl3MatrixView<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    pub(crate) bias: Option<DeviceRange<'a>>,
    pub(crate) shape: [usize; 2],
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}

/// Contiguous F32 activation/output rows and disjoint caller-owned scratch.
/// The owner initializes the sticky flag once and retains it across dependent work.
/// Standalone projection checks health before and after enqueue. The private Flint Frame
/// may defer scratch-only value consumers until its router/final status observations,
/// before any unchecked address/index/geometry decision or external output/state publication.
pub(crate) struct Exl3Io<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) rows: usize,
}

impl Exl3MatrixView<'_> {
    /// Reads F32 directly, before either the q8_1 or tiled fp16 activation pack.
    pub(crate) fn project(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        io: Exl3Io<'_>,
    ) -> Result<(), EngError> {
        context.ensure_usable()?;
        if io.flag.bytes() < 4 {
            return Err(EngError::BadRequest(
                "EXL3 arithmetic flag needs four bytes".into(),
            ));
        }
        // shortcut: host status checks exclude graph capture; use resident status before timing.
        let check_flag = || {
            let mut status = [0u8; 4];
            io.flag
                .buffer()
                .read_at(stream, io.flag.offset(), &mut status)?;
            if u32::from_le_bytes(status) & 1 != 0 {
                return Err(EngError::BadRequest(
                    "EXL3 projection flagged invalid arithmetic".into(),
                ));
            }
            Ok(())
        };
        check_flag()?;
        context.exl3_linear(stream, self.linear_call(&io))?;
        check_flag()
    }

    /// Maps borrowed spans only; enqueueing and observing health remain caller responsibilities.
    pub(in crate::family) fn linear_call<'a>(&'a self, io: &Exl3Io<'a>) -> Exl3LinearCall<'a> {
        Exl3LinearCall {
            trellis: self.trellis,
            suh: self.suh,
            svh: self.svh,
            bias: self.bias,
            x: io.x,
            xh: io.xh,
            z: io.z,
            out: io.out,
            flag: io.flag,
            rows: io.rows,
            split_k: 2,
            k: self.shape[0],
            n: self.shape[1],
            bits_twice: self.bits_twice,
            book: self.book,
        }
    }
}

impl Exl3Weight {
    pub(crate) fn view(&self) -> Result<super::Exl3MatrixView<'_>, EngError> {
        Ok(Exl3MatrixView {
            trellis: self.trellis.range(0, self.trellis.bytes())?,
            suh: self.suh.range(0, self.suh.bytes())?,
            svh: self.svh.range(0, self.svh.bytes())?,
            bias: self
                .bias
                .as_ref()
                .map(|b| b.range(0, b.bytes()))
                .transpose()?,
            shape: self.shape,
            bits_twice: self.bits_twice,
            book: self.book,
        })
    }

    pub(crate) fn project(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        io: Exl3Io<'_>,
    ) -> Result<(), EngError> {
        context.ensure_usable()?;
        self.view()?.project(context, stream, io)
    }

    pub(crate) fn device_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        [
            Some(&self.trellis),
            Some(&self.suh),
            Some(&self.svh),
            self.bias.as_ref(),
        ]
        .into_iter()
        .flatten()
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.device_buffers().map(DeviceBuffer::bytes).sum()
    }
}

impl CudaProjector {
    /// Reads one matrix's stored components; the checkpoint retains metadata, not shard payloads.
    #[cfg(test)]
    pub(crate) fn load_exl3(
        &self,
        checkpoint: &Checkpoint,
        prefix: &str,
        reserve: Reserve,
    ) -> Result<Exl3Weight, EngError> {
        self.context.ensure_usable()?;
        let names: Vec<_> = crate::weights::exl3::COMPONENT_SUFFIXES
            .into_iter()
            .map(|suffix| format!("{prefix}.{suffix}"))
            .collect();
        let keys: Vec<_> = names.iter().map(String::as_str).collect();
        let selected = checkpoint.source.read_selected(&keys, 1 << 30)?;
        let matrix = Matrix::load(&selected, prefix, &checkpoint.config)?;
        self.upload_exl3(&matrix, reserve)
    }

    /// Uploads an already admitted matrix without expanding its packed trellis.
    pub(crate) fn upload_exl3(
        &self,
        matrix: &Matrix,
        reserve: Reserve,
    ) -> Result<Exl3Weight, EngError> {
        self.context.ensure_usable()?;
        let trellis = matrix
            .trellis()
            .host_bytes()
            .ok_or_else(|| EngError::Checkpoint("EXL3 trellis requires host storage".into()))?;
        let suh = float_bytes(matrix.suh())?;
        let svh = float_bytes(matrix.svh())?;
        let bias = matrix.bias().map(float_bytes).transpose()?;
        let bytes = trellis
            .len()
            .checked_add(suh.len())
            .and_then(|bytes| bytes.checked_add(svh.len()))
            .and_then(|bytes| bytes.checked_add(bias.as_ref().map_or(0, Vec::len)))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| EngError::MemoryPlan("EXL3 upload byte count overflow".into()))?;
        self.admit(bytes, reserve)?;
        let trellis = self.context.upload(trellis)?;
        let suh = self.context.upload(&suh)?;
        let svh = self.context.upload(&svh)?;
        let bias = bias
            .as_deref()
            .map(|bytes| self.context.upload(bytes))
            .transpose()?;
        Ok(Exl3Weight {
            bias,
            svh,
            suh,
            trellis,
            shape: matrix.shape(),
            bits_twice: matrix.bitrate_twice(),
            book: matrix.codebook(),
        })
    }
}

fn float_bytes(tensor: &Tensor) -> Result<Vec<u8>, EngError> {
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 scale or bias requires host storage".into()))?;
    let multiplier = match tensor.dtype() {
        Dtype::F16 => 2,
        Dtype::F32 => 1,
        _ => {
            return Err(EngError::Checkpoint(
                "EXL3 scale or bias requires F16 or F32".into(),
            ));
        }
    };
    let output_len = bytes
        .len()
        .checked_mul(multiplier)
        .ok_or_else(|| EngError::MemoryPlan("EXL3 staging byte count overflow".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|error| EngError::MemoryPlan(format!("EXL3 staging allocation: {error}")))?;
    if tensor.dtype() == Dtype::F32 {
        output.extend_from_slice(bytes);
    } else {
        for word in bytes.as_chunks::<2>().0 {
            output.extend_from_slice(
                &f16::from_bits(u16::from_le_bytes(*word))
                    .to_f32()
                    .to_le_bytes(),
            );
        }
    }
    Ok(output)
}

#[cfg(test)]
mod real_projection_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, Tensor};

    #[test]
    fn scale_staging_preserves_half_values_and_float_bias_bits() {
        let bits = [0x0000u16, 0x8000, 0x3c00, 0xc000, 0x0001];
        let scale = Tensor::try_from_host(
            Dtype::F16,
            vec![5],
            bits.into_iter().flat_map(u16::to_le_bytes).collect(),
        )
        .unwrap();
        let expected = [0.0f32, -0.0, 1.0, -2.0, f32::from_bits(0x33800000)];
        assert_eq!(
            float_bytes(&scale).unwrap(),
            expected
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        let raw = [1.25f32, -0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let bias = Tensor::try_from_host(Dtype::F32, vec![2], raw.clone()).unwrap();
        assert_eq!(float_bytes(&bias).unwrap(), raw);
        let invalid = Tensor::try_from_host(Dtype::I16, vec![1], vec![0; 2]).unwrap();
        assert!(matches!(
            float_bytes(&invalid),
            Err(crate::EngError::Checkpoint(_))
        ));
    }
}
