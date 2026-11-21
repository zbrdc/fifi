//! Read-only weight access for the Inkling short-context reference.

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::cuda_projection::{CudaProjection, DenseUpload};

use std::{ops::Range, path::Path};

use super::{TextSpec, bad, checkpoint};
use crate::{
    EngError,
    expert_tier::ExpertId,
    family::mapped_file::MappedFile,
    kernels::{
        nvfp4::E4M3_VALUES,
        nvfp4_cpu::{Activation, Isa, Nvfp4Matrix},
    },
    weights::nemotron_shards::NemotronShardReader as ShardReader,
};

pub(super) struct NativeWeights {
    metadata: ShardReader,
    mapping: MappedFile,
    isa: Isa,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    cuda: Option<CudaProjection>,
}

pub(super) enum Weights {
    Native(Box<NativeWeights>),
    Gguf(Box<super::gguf_weights::GgufTextWeights>),
}

impl Weights {
    pub(super) fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Native(w) => w.vec(name, len),
            Self::Gguf(w) => w.vec(name, len),
        }
    }

    pub(super) fn row(&self, name: &str, row: usize, cols: usize) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Native(w) => w.row(name, row, cols),
            Self::Gguf(w) => w.row(name, row, cols),
        }
    }

    pub(super) fn project(&self, name: &str, x: &[f32], rows: usize) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Native(w) => w.project(name, x, rows),
            Self::Gguf(w) => w.project(name, x, rows),
        }
    }

    pub(super) fn controller(
        &self,
        [down, gain, forget]: [&str; 3],
        x: &[f32],
        [rank, hidden]: [usize; 2],
    ) -> Result<[Vec<f32>; 2], EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = self.cuda_projection()
            && let Some(result) = cuda.controller([down, gain, forget], x, [rank, hidden])
        {
            return result;
        }
        let reduced = self.project(down, x, rank)?;
        Ok([
            self.project(gain, &reduced, hidden)?,
            self.project(forget, &reduced, hidden)?,
        ])
    }

    pub(super) fn activate(&self, x: &[f32]) -> Result<Option<Activation>, EngError> {
        match self {
            Self::Native(w) => w.activate(x).map(Some),
            Self::Gguf(_) => Ok(None),
        }
    }

    pub(super) fn expert(
        &self,
        name: &str,
        id: ExpertId,
        input: Option<&Activation>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Native(w) => w.expert(
                name,
                id.expert as usize,
                input.ok_or_else(|| bad("missing native expert activation"))?,
                x,
            ),
            Self::Gguf(w) => w.expert(name, id, x),
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda(
        &mut self,
        spec: &TextSpec,
        eps: f32,
        host_reserve: u64,
    ) -> Result<(), EngError> {
        match self {
            Self::Native(w) => w.enable_cuda(spec),
            Self::Gguf(w) => w.enable_cuda(spec, eps, host_reserve),
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn cuda_projection(&self) -> Option<&CudaProjection> {
        match self {
            Self::Native(w) => w.cuda_projection(),
            Self::Gguf(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpertPacking {
    Nvfp4,
    Mxfp8,
}

impl ExpertPacking {
    pub(super) fn lengths(self, [rows, cols]: [usize; 2]) -> Option<[usize; 2]> {
        let elements = rows.checked_mul(cols)?;
        let (group, stride) = match self {
            Self::Nvfp4 => (16, 2),
            Self::Mxfp8 => (32, 1),
        };
        (rows > 0 && cols > 0 && cols.is_multiple_of(group))
            .then_some([elements / stride, elements / group])
    }
}

pub(super) struct ExpertMatrix<'a> {
    pub(super) codes: &'a [u8],
    pub(super) scales: &'a [u8],
    pub(super) packing: ExpertPacking,
    pub(super) shape: [usize; 2],
    pub(super) outer: f32,
}

#[derive(Clone, Copy)]
struct Floats<'a> {
    bytes: &'a [u8],
    width: usize,
}

impl Floats<'_> {
    fn len(self) -> usize {
        self.bytes.len() / self.width
    }

    fn values(self) -> impl ExactSizeIterator<Item = f32> {
        self.bytes.chunks_exact(self.width).map(move |bytes| {
            if self.width == 2 {
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
            } else {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
        })
    }

    fn slice(self, range: Range<usize>) -> Option<Self> {
        Some(Self {
            bytes: self
                .bytes
                .get(range.start.checked_mul(self.width)?..range.end.checked_mul(self.width)?)?,
            width: self.width,
        })
    }
}

impl NativeWeights {
    pub(super) fn open(directory: &Path, spec: &TextSpec) -> Result<Self, EngError> {
        let (path, metadata) = checkpoint::open_checkpoint(directory, spec)?;
        let mapping = MappedFile::open(&path, metadata.file_len())
            .map_err(|error| bad(format!("map text weights: {error}")))?;
        Ok(Self {
            metadata,
            mapping,
            isa: Isa::detect(),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            cuda: None,
        })
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda(&mut self, spec: &TextSpec) -> Result<(), EngError> {
        self.cuda = Some(CudaProjection::load(self, spec)?);
        Ok(())
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn dense_uploads(&self, spec: &TextSpec) -> Result<Vec<DenseUpload<'_>>, EngError> {
        let mut uploads = Vec::new();
        for (name, tensor) in self.metadata.tensors() {
            let [rows, cols] = tensor.shape() else {
                continue;
            };
            if tensor.dtype() != "BF16"
                || !name.starts_with("model.llm.")
                || !(name.ends_with("weight") || name.ends_with(".weight_proj"))
                || name == "model.llm.embed.weight"
            {
                continue;
            }
            let rows = if name == "model.llm.unembed.weight" {
                spec.unpadded_vocab_size
            } else {
                *rows
            };
            let shape = [rows as usize, *cols as usize];
            let length = shape[0]
                .checked_mul(shape[1])
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| bad(format!("dense byte count overflows for {name}")))?;
            let codes = self
                .bytes(name)?
                .get(..length)
                .ok_or_else(|| bad(format!("dense projection escapes the tensor {name}")))?;
            uploads.push(DenseUpload { name, codes, shape });
        }
        Ok(uploads)
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn cuda_projection(&self) -> Option<&CudaProjection> {
        self.cuda.as_ref()
    }

    pub(super) fn bytes(&self, name: &str) -> Result<&[u8], EngError> {
        self.metadata
            .absolute_range(name)
            .and_then(|range| self.mapping.as_ref().get(range))
            .ok_or_else(|| bad(format!("missing or out-of-range tensor {name}")))
    }

    fn floats(&self, name: &str) -> Result<Floats<'_>, EngError> {
        let width = match self.metadata.tensor(name).map(|tensor| tensor.dtype()) {
            Some("BF16") => 2,
            Some("F32") => 4,
            _ => return Err(bad(format!("expected floating weights in {name}"))),
        };
        Ok(Floats {
            width,
            bytes: self.bytes(name)?,
        })
    }

    pub(super) fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        if data.len() != len {
            return Err(bad(format!(
                "{name} has {} elements, expected {len}",
                data.len()
            )));
        }
        Ok(data.values().collect())
    }

    pub(super) fn row(&self, name: &str, row: usize, cols: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        let start = row
            .checked_mul(cols)
            .ok_or_else(|| bad("row offset overflows"))?;
        let end = start
            .checked_add(cols)
            .ok_or_else(|| bad("row span overflows"))?;
        data.slice(start..end)
            .map(|range| range.values().collect())
            .ok_or_else(|| bad(format!("{name} row {row} escapes the tensor")))
    }

    pub(super) fn project(&self, name: &str, x: &[f32], rows: usize) -> Result<Vec<f32>, EngError> {
        let data = self.floats(name)?;
        let [stored_rows, stored_cols] = self
            .metadata
            .tensor(name)
            .ok_or_else(|| bad(format!("missing {name}")))?
            .shape()
        else {
            return Err(bad(format!("{name} is not a matrix")));
        };
        if *stored_cols != x.len() as u64 || *stored_rows < rows as u64 || x.is_empty() {
            return Err(bad(format!(
                "{name} projection dimensions disagree with its input or output"
            )));
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = &self.cuda
            && data.width == 2
        {
            return cuda.project(name, [rows, x.len()], x);
        }
        let bytes_per_row = data.width * x.len();
        Ok(data
            .bytes
            .chunks_exact(bytes_per_row)
            .take(rows)
            .map(|bytes| {
                Floats {
                    bytes,
                    width: data.width,
                }
                .values()
                .zip(x)
                .map(|(w, value)| w * value)
                .sum::<f32>()
            })
            .collect())
    }

    pub(super) fn activate(&self, x: &[f32]) -> Result<Activation, EngError> {
        Activation::new(x).map_err(|error| bad(format!("prep expert activation: {error}")))
    }

    pub(super) fn expert(
        &self,
        name: &str,
        expert: usize,
        input: &Activation,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let weights = self.expert_matrix(name, expert)?;
        if x.len() != weights.shape[1] {
            return Err(bad(format!("invalid packed expert input for {name}")));
        }
        if weights.outer == 0.0 {
            return Ok(vec![0.0; weights.shape[0]]);
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = &self.cuda {
            return cuda.expert(name, expert, &weights, x);
        }
        match weights.packing {
            ExpertPacking::Nvfp4 => {
                let [rows, cols] = weights.shape;
                let mut output = vec![0.0; rows];
                Nvfp4Matrix::new(weights.codes, weights.scales, weights.outer, rows, cols)
                    .and_then(|matrix| matrix.matvec_activation(self.isa, input, &mut output))
                    .map_err(|error| bad(format!("{name}: {error}")))?;
                Ok(output)
            }
            ExpertPacking::Mxfp8 => weights
                .codes
                .chunks_exact(x.len())
                .zip(weights.scales.chunks_exact(x.len() / 32))
                .map(|(row, scales)| mxfp8_dot(row, scales, x))
                .collect(),
        }
    }

    pub(super) fn expert_matrix(
        &self,
        name: &str,
        expert: usize,
    ) -> Result<ExpertMatrix<'_>, EngError> {
        let tensor = self
            .metadata
            .tensor(name)
            .ok_or_else(|| bad(format!("missing {name}")))?;
        let [experts, rows, stored_cols] = tensor.shape() else {
            return Err(bad(format!("{name} must have expert, row and column axes")));
        };
        if expert >= *experts as usize {
            return Err(bad(format!("expert {expert} escapes {name}")));
        }
        let (packing, cols, outer) = match tensor.dtype() {
            "U8" => (
                ExpertPacking::Nvfp4,
                stored_cols
                    .checked_mul(2)
                    .ok_or_else(|| bad("packed width overflows"))?,
                self.outer_scale(name, expert)?,
            ),
            "F8_E4M3" => (ExpertPacking::Mxfp8, *stored_cols, 1.0),
            _ => return Err(bad(format!("unsupported expert precision for {name}"))),
        };
        let shape = [*rows as usize, cols as usize];
        let [code_bytes, scale_bytes] = packing
            .lengths(shape)
            .ok_or_else(|| bad(format!("invalid packed expert geometry for {name}")))?;
        Ok(ExpertMatrix {
            codes: self.expert_slice(name, expert, code_bytes)?,
            scales: self.expert_slice(&format!("{name}_scale"), expert, scale_bytes)?,
            packing,
            shape,
            outer,
        })
    }

    fn outer_scale(&self, name: &str, expert: usize) -> Result<f32, EngError> {
        let amax = self.row(&format!("{name}_amax"), expert, 1)?[0];
        let divisor = self.vec(&format!("{name}_outer_scale_denominator"), 1)?[0];
        let scale = amax / divisor;
        if !amax.is_finite()
            || amax < 0.0
            || !divisor.is_finite()
            || divisor <= 0.0
            || !scale.is_finite()
            || (scale == 0.0 && amax != 0.0)
        {
            return Err(bad(format!("invalid NVFP4 outer scale for {name}")));
        }
        Ok(scale)
    }

    fn expert_slice(&self, name: &str, expert: usize, len: usize) -> Result<&[u8], EngError> {
        let start = expert
            .checked_mul(len)
            .ok_or_else(|| bad("expert offset overflows"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| bad("expert span overflows"))?;
        self.bytes(name)?
            .get(start..end)
            .ok_or_else(|| bad(format!("expert {expert} escapes {name}")))
    }
}

pub(super) fn mxfp8_dot(weights: &[u8], scales: &[u8], x: &[f32]) -> Result<f32, EngError> {
    if x.is_empty()
        || weights.len() != x.len()
        || !x.len().is_multiple_of(32)
        || scales.len() != x.len() / 32
    {
        return Err(bad("invalid MXFP8 row geometry"));
    }
    let mut sum = 0.0;
    let (weights, _) = weights.as_chunks::<32>();
    let (x, _) = x.as_chunks::<32>();
    for ((weights, x), &code) in weights.iter().zip(x).zip(scales) {
        if code == 255 {
            return Err(bad("MXFP8 E8M0 scale is NaN"));
        }
        // E8M0 byte zero encodes 2^-127, an F32 subnormal.
        let scale = f32::from_bits(if code == 0 {
            1 << 22
        } else {
            u32::from(code) << 23
        });
        let dot: f32 = weights
            .iter()
            .zip(x)
            .map(|(&weight, value)| E4M3_VALUES[weight as usize] * value)
            .sum();
        sum += scale * dot;
    }
    if !sum.is_finite() {
        return Err(bad("non-finite MXFP8 projection"));
    }
    Ok(sum)
}

#[test]
fn mxfp8_dot_applies_block_exponents_without_expanding_experts() {
    let mut weights = [0x38; 64]; // E4M3 1.0
    weights[32..].fill(0xb8); // E4M3 -1.0
    let x = [1.0; 64];
    assert_eq!(mxfp8_dot(&weights, &[127, 126], &x).unwrap(), 16.0);
    assert!(mxfp8_dot(&weights, &[127, 255], &x).is_err());
    assert!(mxfp8_dot(&weights, &[127], &x).is_err());
}
