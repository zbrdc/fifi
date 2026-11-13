//! The record of one routed expert in a Flash-Next slot bank, and the arithmetic of a bank of
//! records.
//!
//! WHAT: an expert's three matrices (gate, up, down) as one run of bytes, parts in that order and
//! each part as `trellis | suh | svh` with `suh` and `svh` widened to F32, which is how the EXL3
//! kernels and the CPU kernel read them. A slot of a bank is one record; the host store keeps the
//! records of every expert of a layer in the same form. WHY one record: the id-indexed EXL3 launch
//! walks its members at a fixed byte stride, so with equal strides for the trellis (bytes) and the
//! scales (floats), the slot index that `tiered_classify` writes addresses all three components
//! without an offset table, and a miss reads slot 0 with weight +0.0. A host record is also a
//! ready `ExpertView` for the CPU kernel, so a cold expert is served from the same bytes that
//! a copy would move to the device.
//!
//! Each launch of a part uses three ranges of the bank buffer, one per component, starting at that
//! part's offset in record 0 and extended over all slots ([`SlotBankLayout::bank_ranges`]).
//!
//! Nothing here allocates device memory or changes how any model loads: Qwen3.6's EXL3 bank
//! (`upload_exl3_bank`, one record buffer per part) is untouched, and only the Flash-Next
//! step will build a slot bank.
//!
//! Pure host code.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::ops::Range;

use half::f16;

use crate::{
    EngError,
    kernels::exl3_cpu::{Exl3Projection, ExpertView},
    tensor::Dtype,
    weights::exl3::Matrix,
};

/// Parts of an expert: gate, up, down.
pub(in crate::family::flash_next) const PARTS: usize = 3;
const F32: usize = 4;
/// Both matrix sides must be multiples of the EXL3 block (the CPU kernel's and the launches' grid).
const BLOCK: usize = 128;

fn refuse(detail: String) -> EngError {
    EngError::Checkpoint(format!("Flash-Next slot bank: {detail}"))
}

/// Words (u16) of one 16 x 16 trellis tile at a rate: `bits_twice * 8`. The CPU kernel takes
/// the integer rates 1 to 8 and the half-bit rates 24, 40 and 56.
fn supported_words(words: usize) -> bool {
    (words.is_multiple_of(16) && (16..=128).contains(&words)) || matches!(words, 24 | 40 | 56)
}

/// Where the components of one part sit in a record, as byte ranges from the record start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct PartLayout {
    /// `[k, n]` of the matrix: the input and the output width.
    pub(in crate::family::flash_next) shape: [usize; 2],
    /// u16 words per tile.
    pub(in crate::family::flash_next) words: usize,
    pub(in crate::family::flash_next) trellis: Range<usize>,
    pub(in crate::family::flash_next) suh: Range<usize>,
    pub(in crate::family::flash_next) svh: Range<usize>,
}

/// The bytes of one part as the checkpoint gives them, `suh` and `svh` already F32.
#[derive(Debug, Clone, Copy)]
pub(in crate::family::flash_next) struct PartBytes<'a> {
    pub(in crate::family::flash_next) trellis: &'a [u8],
    pub(in crate::family::flash_next) suh: &'a [u8],
    pub(in crate::family::flash_next) svh: &'a [u8],
}

/// An owned part, widened from a stored matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct OwnedPart {
    pub(in crate::family::flash_next) shape: [usize; 2],
    pub(in crate::family::flash_next) words: usize,
    pub(in crate::family::flash_next) trellis: Vec<u8>,
    pub(in crate::family::flash_next) suh: Vec<u8>,
    pub(in crate::family::flash_next) svh: Vec<u8>,
}

impl OwnedPart {
    /// The trellis as stored and the scales widened to F32 (an F16 scale widens exactly).
    pub(in crate::family::flash_next) fn from_matrix(matrix: &Matrix) -> Result<Self, EngError> {
        let widen = |tensor: &crate::tensor::Tensor| -> Result<Vec<u8>, EngError> {
            let bytes = tensor
                .host_bytes()
                .ok_or_else(|| refuse("a scale without host storage".into()))?;
            match tensor.dtype() {
                Dtype::F32 => Ok(bytes.to_vec()),
                Dtype::F16 => Ok(bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|pair| f16::from_le_bytes(*pair).to_f32().to_le_bytes())
                    .collect()),
                other => Err(refuse(format!("a scale stored as {other:?}"))),
            }
        };
        Ok(Self {
            shape: matrix.shape(),
            words: usize::from(matrix.bitrate_twice()) * 8,
            trellis: matrix
                .trellis()
                .host_bytes()
                .ok_or_else(|| refuse("a trellis without host storage".into()))?
                .to_vec(),
            suh: widen(matrix.suh())?,
            svh: widen(matrix.svh())?,
        })
    }

    pub(in crate::family::flash_next) fn bytes(&self) -> PartBytes<'_> {
        PartBytes {
            trellis: &self.trellis,
            suh: &self.suh,
            svh: &self.svh,
        }
    }
}

/// The record of one expert: three parts back to back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct SlotBankLayout {
    parts: [PartLayout; PARTS],
    record_bytes: usize,
}

impl SlotBankLayout {
    /// `hidden` and `inter` are the model's widths (gate and up are `[hidden, inter]`, down is
    /// `[inter, hidden]`); `words` is the tile size of each part. Refuses a width off the 128
    /// grid, an unsupported rate and a size that overflows.
    pub(in crate::family::flash_next) fn new(
        hidden: usize,
        inter: usize,
        words: [usize; PARTS],
    ) -> Result<Self, EngError> {
        for (name, width) in [("hidden", hidden), ("inter", inter)] {
            if width == 0 || !width.is_multiple_of(BLOCK) {
                return Err(refuse(format!(
                    "{name} {width} is not a nonzero multiple of {BLOCK}"
                )));
            }
        }
        if let Some(bad) = words.iter().find(|&&w| !supported_words(w)) {
            return Err(refuse(format!("{bad} words per tile is not a stored rate")));
        }
        let overflow = || refuse("the record size overflows".into());
        let shapes = [[hidden, inter], [hidden, inter], [inter, hidden]];
        let mut at = 0usize;
        let mut parts = Vec::with_capacity(PARTS);
        for (shape, words) in shapes.into_iter().zip(words) {
            let [k, n] = shape;
            let tiles = (k / 16).checked_mul(n / 16).ok_or_else(overflow)?;
            let trellis_end = tiles
                .checked_mul(words * 2)
                .and_then(|bytes| at.checked_add(bytes))
                .ok_or_else(overflow)?;
            let suh_end = k
                .checked_mul(F32)
                .and_then(|bytes| trellis_end.checked_add(bytes))
                .ok_or_else(overflow)?;
            let svh_end = n
                .checked_mul(F32)
                .and_then(|bytes| suh_end.checked_add(bytes))
                .ok_or_else(overflow)?;
            parts.push(PartLayout {
                shape,
                words,
                trellis: at..trellis_end,
                suh: trellis_end..suh_end,
                svh: suh_end..svh_end,
            });
            at = svh_end;
        }
        let parts: [PartLayout; PARTS] = parts
            .try_into()
            .map_err(|_| refuse("three parts expected".into()))?;
        Ok(Self {
            parts,
            record_bytes: at,
        })
    }

    pub(in crate::family::flash_next) fn part(&self, index: usize) -> &PartLayout {
        &self.parts[index]
    }

    /// Bytes of one record, which is also the byte stride between slots.
    pub(in crate::family::flash_next) fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    /// The stride between the trellises of consecutive slots, in bytes.
    pub(in crate::family::flash_next) fn trellis_stride_bytes(&self) -> usize {
        self.record_bytes
    }

    /// The stride between the `suh` (and the `svh`) of consecutive slots, in floats. Every part
    /// is a multiple of four bytes, so the record is too, and the byte strides of all three
    /// components are equal.
    pub(in crate::family::flash_next) fn scale_stride_floats(&self) -> usize {
        self.record_bytes / F32
    }

    /// Byte offset of slot `slot` in a bank buffer.
    pub(in crate::family::flash_next) fn slot_offset(
        &self,
        slot: usize,
    ) -> Result<usize, EngError> {
        slot.checked_mul(self.record_bytes)
            .ok_or_else(|| refuse(format!("slot {slot} overflows the buffer offset")))
    }

    /// Bytes of a bank buffer of `slots` slots.
    pub(in crate::family::flash_next) fn bank_bytes(
        &self,
        slots: usize,
    ) -> Result<usize, EngError> {
        self.slot_offset(slots)
    }

    /// The three component ranges of part `part` for a launch over `slots` slots of a bank buffer:
    /// each starts at the component's place in record 0 and ends at its place in the last record,
    /// so member `i` of the launch is the component at `start + i * stride`.
    pub(in crate::family::flash_next) fn bank_ranges(
        &self,
        part: usize,
        slots: usize,
    ) -> Result<[Range<usize>; 3], EngError> {
        if slots == 0 {
            return Err(refuse("a bank needs at least one slot".into()));
        }
        let layout = self
            .parts
            .get(part)
            .ok_or_else(|| refuse(format!("part {part} is not gate, up or down")))?;
        let last = self.slot_offset(slots - 1)?;
        let span = |range: &Range<usize>| range.start..range.end + last;
        Ok([span(&layout.trellis), span(&layout.suh), span(&layout.svh)])
    }

    /// Writes one expert into `out` (exactly one record). Refuses a part whose component lengths
    /// or shape do not match this layout; nothing is written then.
    pub(in crate::family::flash_next) fn pack_into(
        &self,
        parts: [PartBytes<'_>; PARTS],
        out: &mut [u8],
    ) -> Result<(), EngError> {
        if out.len() != self.record_bytes {
            return Err(refuse(format!(
                "a record is {} bytes, the buffer has {}",
                self.record_bytes,
                out.len()
            )));
        }
        for (index, (layout, part)) in self.parts.iter().zip(&parts).enumerate() {
            for (name, range, bytes) in [
                ("trellis", &layout.trellis, part.trellis),
                ("suh", &layout.suh, part.suh),
                ("svh", &layout.svh, part.svh),
            ] {
                if bytes.len() != range.len() {
                    return Err(refuse(format!(
                        "part {index} {name} is {} bytes, the layout has {}",
                        bytes.len(),
                        range.len()
                    )));
                }
            }
        }
        for (layout, part) in self.parts.iter().zip(parts) {
            out[layout.trellis.clone()].copy_from_slice(part.trellis);
            out[layout.suh.clone()].copy_from_slice(part.suh);
            out[layout.svh.clone()].copy_from_slice(part.svh);
        }
        Ok(())
    }

    /// One expert's matrices as a record. Refuses a matrix of another shape or rate than the layout.
    pub(in crate::family::flash_next) fn pack_matrices(
        &self,
        matrices: [&Matrix; PARTS],
    ) -> Result<Vec<u8>, EngError> {
        let owned = [
            OwnedPart::from_matrix(matrices[0])?,
            OwnedPart::from_matrix(matrices[1])?,
            OwnedPart::from_matrix(matrices[2])?,
        ];
        for (index, (layout, part)) in self.parts.iter().zip(&owned).enumerate() {
            if part.shape != layout.shape || part.words != layout.words {
                return Err(refuse(format!(
                    "part {index} is {:?} at {} words, the layout is {:?} at {}",
                    part.shape, part.words, layout.shape, layout.words
                )));
            }
        }
        let mut record = vec![0u8; self.record_bytes];
        self.pack_into(
            [owned[0].bytes(), owned[1].bytes(), owned[2].bytes()],
            &mut record,
        )?;
        Ok(record)
    }

    /// The CPU kernel's view of the expert in `record`.
    pub(in crate::family::flash_next) fn view<'a>(
        &self,
        record: &'a [u8],
    ) -> Result<ExpertView<'a>, EngError> {
        if record.len() != self.record_bytes {
            return Err(refuse(format!(
                "a record is {} bytes, received {}",
                self.record_bytes,
                record.len()
            )));
        }
        let projection = |index: usize| {
            let layout = &self.parts[index];
            Exl3Projection::new(
                layout.shape,
                layout.words,
                &record[layout.trellis.clone()],
                &record[layout.suh.clone()],
                &record[layout.svh.clone()],
            )
        };
        Ok(ExpertView {
            gate: projection(0)?,
            up: projection(1)?,
            down: projection(2)?,
        })
    }
}

#[cfg(test)]
mod tests;
