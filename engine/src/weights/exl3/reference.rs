use half::f16;

use super::{MCG_MARKER, MUL1_MARKER};

use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codebook {
    ThreeInst,
    Mcg,
    Mul1,
}

impl Codebook {
    /// Decodes a tail-biting state to its IEEE binary16 bits.
    pub fn decode_state(self, state: u16) -> u16 {
        let state = u32::from(state);
        let x = match self {
            Self::ThreeInst => state.wrapping_mul(89_226_354).wrapping_add(64_248_484),
            Self::Mcg => state.wrapping_mul(MCG_MARKER),
            Self::Mul1 => {
                let x = state.wrapping_mul(MUL1_MARKER);
                let sum: f64 = x.to_le_bytes().into_iter().map(f64::from).sum();
                let offset = f16::from_bits(0x6400).to_f64();
                let scale = f16::from_bits(0x1eee).to_f64();
                let bias = f16::from_bits(0xc931).to_f64();
                // These intermediates are exact in f64; round once, as an fp16 FMA does.
                return f16::from_f64((offset + sum) * scale + bias).to_bits();
            }
        };
        let x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
        let low = f16::from_bits(x as u16).to_f64();
        let high = f16::from_bits((x >> 16) as u16).to_f64();
        f16::from_f64(low + high).to_bits()
    }

    /// Reconstructs one 16×16 W' tile, with input channels on the first axis.
    pub fn decode_tile(self, trellis: &[u16], output: &mut [u16; 256]) -> Result<(), EngError> {
        let bits_twice = self.bits_twice(trellis.len())?;
        let mut stream = [0u32; 64];
        for (word, pair) in stream.iter_mut().zip(trellis.as_chunks::<2>().0.iter()) {
            *word = u32::from(pair[0]) | (u32::from(pair[1]) << 16);
        }
        let stream = &stream[..trellis.len() / 2];
        for p in 0..256 {
            let end = (p + 1) * bits_twice / 2;
            let state = stream_state(stream, end);
            let lane = p >> 3;
            let j = p & 7;
            let row = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
            let col = (lane >> 2) + 8 * (j >> 2);
            output[row * 16 + col] = self.decode_state(state);
        }
        Ok(())
    }

    /// Writes unrotated W' as input-major [k, n] binary16 bits into caller-owned storage.
    pub fn reconstruct_into(self, trellis: &Tensor, output: &mut [u16]) -> Result<(), EngError> {
        let (k, n, words) = matrix_shape(self, trellis)?;
        let expected = k
            .checked_mul(n)
            .ok_or_else(|| EngError::Checkpoint("EXL3 matrix extent exceeds host limits".into()))?;
        if output.len() != expected {
            return Err(EngError::BadRequest(format!(
                "EXL3 output requires {expected} elements, received {}",
                output.len()
            )));
        }
        let bytes = trellis.host_bytes().ok_or_else(|| {
            EngError::Checkpoint("EXL3 reference requires host trellis storage".into())
        })?;
        let mut tile_words = [0u16; 128];
        let mut tile = [0u16; 256];
        let n_tiles = n / 16;
        for (index, bytes) in bytes.chunks_exact(words * 2).enumerate() {
            for (slot, pair) in tile_words.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                *slot = u16::from_le_bytes([pair[0], pair[1]]);
            }
            self.decode_tile(&tile_words[..words], &mut tile)?;
            let row = (index / n_tiles) * 16;
            let col = (index % n_tiles) * 16;
            for (local_row, values) in tile.as_chunks::<16>().0.iter().enumerate() {
                let start = (row + local_row) * n + col;
                output[start..start + 16].copy_from_slice(values);
            }
        }
        Ok(())
    }

    fn bits_twice(self, words: usize) -> Result<usize, EngError> {
        let integer = matches!(words, 16 | 32 | 48 | 64 | 80 | 96 | 112 | 128);
        let half = self == Self::Mul1 && matches!(words, 24 | 40 | 56);
        if !integer && !half {
            return Err(EngError::Checkpoint(format!(
                "EXL3 {self:?} does not support {words} words per tile"
            )));
        }
        Ok(words / 8)
    }
}

pub(super) fn matrix_shape(
    book: Codebook,
    trellis: &Tensor,
) -> Result<(usize, usize, usize), EngError> {
    if trellis.dtype() != Dtype::I16 {
        return Err(EngError::Checkpoint(
            "EXL3 trellis storage must be I16".into(),
        ));
    }
    let [k_tiles, n_tiles, words] = trellis.shape() else {
        return Err(EngError::Checkpoint(
            "EXL3 trellis must have rank three".into(),
        ));
    };
    if *k_tiles == 0 || *n_tiles == 0 || !k_tiles.is_multiple_of(8) || !n_tiles.is_multiple_of(8) {
        return Err(EngError::Checkpoint(
            "EXL3 dimensions must be nonzero multiples of 128".into(),
        ));
    }
    let words = usize::try_from(*words)
        .map_err(|_| EngError::Checkpoint("EXL3 tile extent exceeds host limits".into()))?;
    book.bits_twice(words)?;
    let extent = |tiles: u32| {
        usize::try_from(tiles)
            .ok()
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| EngError::Checkpoint("EXL3 matrix extent exceeds host limits".into()))
    };
    Ok((extent(*k_tiles)?, extent(*n_tiles)?, words))
}

fn stream_state(words: &[u32], end: usize) -> u16 {
    let bits = words.len() * 32;
    let start = (end + bits - 16) % bits;
    let index = start / 32;
    let pair = (u64::from(words[index]) << 32) | u64::from(words[(index + 1) % words.len()]);
    (pair >> (48 - start % 32)) as u16
}
