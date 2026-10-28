use half::f16;

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
    weights::WeightSource,
};

use super::{Codebook, Config, MCG_MARKER, MUL1_MARKER, reference::matrix_shape};

/// An admitted stored matrix; neither family dispatch nor execution is selected here.
#[derive(Debug, Clone)]
pub struct Matrix {
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    bias: Option<Tensor>,
    book: Codebook,
    shape: [usize; 2],
    bits_twice: u8,
}

impl Matrix {
    /// Loads a tensor group by its prefix, retaining shared backing across shards.
    pub fn load(
        source: &dyn WeightSource,
        prefix: &str,
        config: &Config,
    ) -> Result<Self, EngError> {
        for suffix in ["su", "sv", "scale"] {
            if source.tensor(&format!("{prefix}.{suffix}")).is_some() {
                return Err(invalid(
                    "old packed signs or a separate scale are unsupported",
                ));
            }
        }
        let trellis = required(source, prefix, "trellis")?;
        let suh = required(source, prefix, "suh")?;
        let svh = required(source, prefix, "svh")?;
        let mcg = source.tensor(&format!("{prefix}.mcg"));
        let mul1 = source.tensor(&format!("{prefix}.mul1"));
        let book = match (mcg, mul1) {
            (None, None) => Codebook::ThreeInst,
            (Some(marker), None) => {
                check_marker(&marker, MCG_MARKER)?;
                Codebook::Mcg
            }
            (None, Some(marker)) => {
                check_marker(&marker, MUL1_MARKER)?;
                Codebook::Mul1
            }
            (Some(_), Some(_)) => return Err(invalid("multiple codebook markers")),
        };
        if config.codebook.is_some_and(|declared| declared != book) {
            return Err(invalid("codebook marker disagrees with config"));
        }
        let (k, n, words) = matrix_shape(book, &trellis)?;
        check_vector(&suh, k, &[Dtype::F16], "suh")?;
        check_vector(&svh, n, &[Dtype::F16], "svh")?;
        let bias = source.tensor(&format!("{prefix}.bias"));
        if let Some(bias) = &bias {
            check_vector(bias, n, &[Dtype::F16, Dtype::F32], "bias")?;
        }
        Ok(Self {
            trellis,
            suh,
            svh,
            bias,
            book,
            shape: [k, n],
            bits_twice: (words / 8) as u8,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    /// Twice the physical bits per weight: 6 means 3.0 bits per weight.
    pub fn bitrate_twice(&self) -> u8 {
        self.bits_twice
    }
    pub fn codebook(&self) -> Codebook {
        self.book
    }
    pub fn trellis(&self) -> &Tensor {
        &self.trellis
    }
    pub fn suh(&self) -> &Tensor {
        &self.suh
    }
    pub fn svh(&self) -> &Tensor {
        &self.svh
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

fn required(source: &dyn WeightSource, prefix: &str, suffix: &str) -> Result<Tensor, EngError> {
    let name = format!("{prefix}.{suffix}");
    source
        .tensor(&name)
        .ok_or_else(|| invalid(&format!("missing tensor {name:?}")))
}

fn check_marker(marker: &Tensor, expected: u32) -> Result<(), EngError> {
    if marker.dtype() != Dtype::I32 || !marker.shape().is_empty() {
        return Err(invalid("codebook marker must be scalar I32"));
    }
    let bytes = marker
        .host_bytes()
        .and_then(|v| <[u8; 4]>::try_from(v).ok())
        .ok_or_else(|| invalid("codebook marker requires four host bytes"))?;
    if u32::from_le_bytes(bytes) != expected {
        return Err(invalid("unknown codebook marker word"));
    }
    Ok(())
}

fn check_vector(
    tensor: &Tensor,
    length: usize,
    dtypes: &[Dtype],
    label: &str,
) -> Result<(), EngError> {
    if !dtypes.contains(&tensor.dtype())
        || tensor.shape().len() != 1
        || tensor.shape()[0] as usize != length
    {
        return Err(invalid(&format!("{label} has an invalid dtype or shape")));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| invalid(&format!("{label} requires host storage")))?;
    let finite = match tensor.dtype() {
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| f16::from_bits(u16::from_le_bytes(*v)).is_finite()),
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite()),
        _ => false,
    };
    if !finite {
        return Err(invalid(&format!("{label} contains nonfinite values")));
    }
    Ok(())
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 matrix: {message}"))
}
