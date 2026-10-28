use crate::EngError;
use crate::tensor::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentLayout {
    pub heads: u32,
    pub rank: u32,
    pub conv_dim: u32,
    pub conv_width: u32,
    pub dtype: Dtype,
}

impl RecurrentLayout {
    fn checked(&self) -> Result<Self, EngError> {
        for (field, value) in [
            ("heads", self.heads),
            ("rank", self.rank),
            ("conv_dim", self.conv_dim),
            ("conv_width", self.conv_width),
        ] {
            if value == 0 {
                return Err(EngError::Checkpoint(format!(
                    "recurrent layout {field} must exceed zero"
                )));
            }
        }
        Ok(*self)
    }

    pub fn conv_shape(&self) -> Result<[u32; 2], EngError> {
        let layout = self.checked()?;
        let width = layout
            .conv_width
            .checked_sub(1)
            .expect("conv_width exceeds zero after the layout check");
        Ok([layout.conv_dim, width])
    }

    pub fn ssm_shape(&self) -> Result<[u32; 3], EngError> {
        let layout = self.checked()?;
        Ok([layout.heads, layout.rank, layout.rank])
    }

    pub fn conv_bytes(&self) -> Result<u64, EngError> {
        let [dim, width] = self.conv_shape()?;
        shape_bytes(&[dim, width], self.dtype)
    }

    pub fn ssm_bytes(&self) -> Result<u64, EngError> {
        let shape = self.ssm_shape()?;
        shape_bytes(&shape, self.dtype)
    }

    pub fn byte_len(&self) -> Result<u64, EngError> {
        let conv = self.conv_bytes()?;
        let ssm = self.ssm_bytes()?;
        conv.checked_add(ssm)
            .ok_or_else(|| EngError::Checkpoint("recurrent state byte count overflows u64".into()))
    }
}

fn shape_bytes(shape: &[u32], dtype: Dtype) -> Result<u64, EngError> {
    let elements = shape
        .iter()
        .try_fold(1u64, |product, dim| product.checked_mul(u64::from(*dim)));
    let elements = elements
        .ok_or_else(|| EngError::Checkpoint("recurrent state shape overflows u64".into()))?;
    elements
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| EngError::Checkpoint("recurrent state byte count overflows u64".into()))
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use crate::EngError;
    use crate::kernels::MlxKernels;
    use crate::tensor::{Dtype, ExecutionTensor, Tensor};

    use super::{RecurrentLayout, shape_bytes};

    pub struct RecurrentState {
        conv: ExecutionTensor,
        ssm: ExecutionTensor,
        layout: RecurrentLayout,
    }

    impl RecurrentState {
        pub fn zeros(kernels: &mut MlxKernels, layout: &RecurrentLayout) -> Result<Self, EngError> {
            let layout = layout.checked()?;
            let conv = Self::import_zeros(kernels, &layout.conv_shape()?, layout.dtype)?;
            let ssm = Self::import_zeros(kernels, &layout.ssm_shape()?, layout.dtype)?;
            Ok(Self { conv, ssm, layout })
        }

        fn import_zeros(
            kernels: &mut MlxKernels,
            shape: &[u32],
            dtype: Dtype,
        ) -> Result<ExecutionTensor, EngError> {
            let bytes = shape_bytes(shape, dtype)?;
            let len = usize::try_from(bytes)
                .map_err(|_| EngError::Checkpoint("recurrent state exceeds usize".into()))?;
            let host = Tensor::try_from_host(dtype, shape.to_vec(), vec![0u8; len])?;
            kernels.import_tensor(&host)
        }

        pub fn conv(&self) -> &ExecutionTensor {
            &self.conv
        }

        pub fn ssm(&self) -> &ExecutionTensor {
            &self.ssm
        }

        pub fn layout(&self) -> RecurrentLayout {
            self.layout
        }

        pub fn owner_key(&self) -> crate::tensor::OwnerKey {
            self.conv.owner_key()
        }
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub use native::RecurrentState;
