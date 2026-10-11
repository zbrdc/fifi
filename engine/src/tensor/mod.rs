use std::{
    ops::{Deref, Range},
    sync::Arc,
};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use std::{num::NonZeroU64, rc::Rc};

use crate::EngError;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::kernels::mlx::native::Resident;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerKey(pub(crate) NonZeroU64);

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub struct ExecutionTensor {
    pub(crate) storage: Rc<Resident>,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl ExecutionTensor {
    pub(crate) fn new(storage: Resident) -> Self {
        Self {
            storage: Rc::new(storage),
        }
    }

    pub fn dtype(&self) -> Dtype {
        self.storage.dtype()
    }

    pub fn shape(&self) -> &[u32] {
        self.storage.shape()
    }

    pub fn byte_len(&self) -> u64 {
        self.storage.byte_len()
    }

    pub fn share(&self) -> Self {
        Self {
            storage: Rc::clone(&self.storage),
        }
    }

    pub(crate) fn owner_key(&self) -> OwnerKey {
        self.storage.owner_key()
    }

    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.storage, &other.storage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    U8,
    U32,
    I64,
}

impl Dtype {
    /// Stored bytes per element. The family code, the kernel validation code, and the native
    /// adapter all use this one table.
    pub(crate) fn byte_width(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::U32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Backing {
    Bytes(Arc<[u8]>),
    Mapped(Arc<memmap2::Mmap>),
}

impl Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Mapped(bytes) => bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    dtype: Dtype,
    shape: Vec<u32>,
    bytes: Backing,
    range: Range<usize>,
    byte_len: u64,
}

impl Tensor {
    pub fn try_from_host(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Result<Self, EngError> {
        let len = bytes.len();
        Self::from_shared(dtype, shape, Backing::Bytes(Arc::from(bytes)), 0..len)
    }

    pub(crate) fn from_shared(
        dtype: Dtype,
        shape: Vec<u32>,
        bytes: Backing,
        range: Range<usize>,
    ) -> Result<Self, EngError> {
        let elements = shape.iter().try_fold(1u64, |count, &dim| {
            count.checked_mul(u64::from(dim)).ok_or_else(|| {
                EngError::Checkpoint("stored dimension product overflows u64".into())
            })
        })?;
        let byte_len = elements
            .checked_mul(dtype.byte_width())
            .ok_or_else(|| EngError::Checkpoint("stored byte length overflows u64".into()))?;
        let expected = usize::try_from(byte_len)
            .map_err(|_| EngError::Checkpoint("stored byte length exceeds host limits".into()))?;
        let payload = bytes
            .get(range.clone())
            .ok_or_else(|| EngError::Checkpoint("tensor range exceeds backing bytes".into()))?;
        if payload.len() != expected {
            return Err(EngError::Checkpoint(format!(
                "stored shape requires {expected} bytes, received {}",
                payload.len()
            )));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
            range,
            byte_len,
        })
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn host_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes[self.range.clone()])
    }
}
