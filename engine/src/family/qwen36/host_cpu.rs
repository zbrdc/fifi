//! CPU projector for the Q36 host reference: affine packed weights multiplied on the host.
//!
//! The reference run must not share a projection kernel with the resident decoder, so this adapter
//! reads the packed words itself. It supports affine weights at 2, 4, and 8 bits with BF16 or F32
//! scales. Element `c` of a row sits in word `c / (32 / bits)` at bit offset `(c % (32 / bits)) * bits`
//! (least significant bits first) and has the value `scale[g] * q + bias[g]` for group `g = c / group`.

use crate::EngError;
use crate::family::flash_next::stored::{self, StoredWeight};
use crate::family::project::Projector;
use crate::kernels::{PackedWeight, QuantizationMode};
use crate::tensor::{Dtype, Tensor};
use crate::weights::StreamedCheckpoint;

/// Borrowed bytes of one packed matrix `[rows, cols]`.
struct View<'a> {
    words: &'a [u8],
    scales: &'a [u8],
    biases: &'a [u8],
    scale_dtype: Dtype,
    rows: usize,
    cols: usize,
    bits: u32,
    group: usize,
}

fn unsupported(message: &str) -> EngError {
    EngError::BadRequest(format!("cpu projector: {message}"))
}

/// The host bytes of an optional tensor.
fn host_bytes(tensor: Option<&Tensor>) -> Result<&[u8], EngError> {
    tensor
        .and_then(Tensor::host_bytes)
        .ok_or_else(|| unsupported("packed weight bytes are not on the host"))
}

impl<'a> View<'a> {
    fn new(weight: &'a PackedWeight) -> Result<Self, EngError> {
        let spec = weight.quantization();
        if spec.mode != QuantizationMode::Affine || !matches!(spec.bits, 2 | 4 | 8) {
            return Err(unsupported(
                "only affine weights at 2, 4, or 8 bits are supported",
            ));
        }
        let &[rows, cols] = weight.logical_shape() else {
            return Err(unsupported("a packed matrix must have rank 2"));
        };
        let scale_dtype = weight.scales().dtype();
        if !matches!(scale_dtype, Dtype::Bf16 | Dtype::F32) {
            return Err(unsupported("scales must be BF16 or F32"));
        }
        Ok(Self {
            words: host_bytes(Some(weight.payload()))?,
            scales: host_bytes(Some(weight.scales()))?,
            biases: host_bytes(weight.biases())?,
            scale_dtype,
            rows: rows as usize,
            cols: cols as usize,
            bits: spec.bits,
            group: spec.group_size as usize,
        })
    }

    fn real(&self, bytes: &[u8], index: usize) -> f32 {
        match self.scale_dtype {
            Dtype::Bf16 => {
                let at = index * 2;
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])) << 16)
            }
            _ => {
                let at = index * 4;
                f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
            }
        }
    }

    /// The code `q` of element `col` in row `row`.
    fn code(&self, row: usize, col: usize) -> u32 {
        let per_word = (32 / self.bits) as usize;
        let words_per_row = self.cols / per_word;
        let at = (row * words_per_row + col / per_word) * 4;
        let word = u32::from_le_bytes([
            self.words[at],
            self.words[at + 1],
            self.words[at + 2],
            self.words[at + 3],
        ]);
        (word >> ((col % per_word) as u32 * self.bits)) & ((1u32 << self.bits) - 1)
    }

    /// `sum_c x[c] * (scale * q + bias)` for one row; `group_sums[g]` is the sum of `x` over group `g`.
    fn dot_row(&self, row: usize, x: &[f32], group_sums: &[f32]) -> f32 {
        let groups = self.cols / self.group;
        let mut total = 0.0f64;
        for (g, (part, sum)) in x.chunks(self.group).zip(group_sums).enumerate() {
            let first = g * self.group;
            let weighted: f32 = part
                .iter()
                .enumerate()
                .map(|(offset, value)| value * self.code(row, first + offset) as f32)
                .sum();
            let scale = self.real(self.scales, row * groups + g);
            let bias = self.real(self.biases, row * groups + g);
            total += f64::from(scale * weighted + bias * sum);
        }
        total as f32
    }

    /// The dequantized values of one row.
    fn dequantize_row(&self, row: usize) -> Vec<f32> {
        let groups = self.cols / self.group;
        (0..self.cols)
            .map(|col| {
                let at = row * groups + col / self.group;
                self.real(self.scales, at) * self.code(row, col) as f32 + self.real(self.biases, at)
            })
            .collect()
    }
}

/// Dequantizes a whole packed matrix `[rows, cols]` to row-major `f32` by the rule of [`View`]:
/// `scale[g] * q + bias[g]`. The resident loader uses it for the small router and gate matrices.
pub(super) fn dequantize_matrix(weight: &PackedWeight) -> Result<Vec<f32>, EngError> {
    let view = View::new(weight)?;
    Ok((0..view.rows)
        .flat_map(|row| view.dequantize_row(row))
        .collect())
}

/// `x * W^T` for a packed matrix `[rows, cols]`, rows split over `threads` threads.
fn matvec(weight: &PackedWeight, x: &[f32], threads: usize) -> Result<Vec<f32>, EngError> {
    let view = View::new(weight)?;
    if x.len() != view.cols {
        return Err(EngError::BadRequest(format!(
            "projection input has {} values, expected {}",
            x.len(),
            view.cols
        )));
    }
    let group_sums: Vec<f32> = x.chunks(view.group).map(|part| part.iter().sum()).collect();
    let mut out = vec![0.0f32; view.rows];
    let chunk = view.rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (index, part) in out.chunks_mut(chunk).enumerate() {
            let (view, group_sums) = (&view, &group_sums);
            scope.spawn(move || {
                for (offset, slot) in part.iter_mut().enumerate() {
                    *slot = view.dot_row(index * chunk + offset, x, group_sums);
                }
            });
        }
    });
    Ok(out)
}

/// Executes projections on the CPU over stored weights. Expert stacks and the embedding table are
/// read from the checkpoint one slice at a time.
pub(crate) struct CpuProjector<'c> {
    ckpt: &'c StreamedCheckpoint,
    threads: usize,
}

impl<'c> CpuProjector<'c> {
    /// Creates a projector that splits matrix rows over `threads` threads (at least one).
    pub(crate) fn new(ckpt: &'c StreamedCheckpoint, threads: usize) -> Self {
        Self { ckpt, threads }
    }
}

impl Projector for CpuProjector<'_> {
    type Weight = StoredWeight;

    fn linear(&mut self, weight: &StoredWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        match weight {
            StoredWeight::Packed(packed) => matvec(packed, x, self.threads),
            StoredWeight::Sliced(_) => {
                Err(EngError::BadRequest("linear needs a packed matrix".into()))
            }
        }
    }

    fn expert_linear(
        &mut self,
        stack: &StoredWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match stack {
            StoredWeight::Sliced(slices) => {
                let packed = stored::read_slice(self.ckpt, slices, expert)?;
                matvec(&packed, x, self.threads)
            }
            StoredWeight::Packed(_) => Err(EngError::BadRequest(
                "expert_linear needs a sliced stack".into(),
            )),
        }
    }

    fn rows(&mut self, table: &StoredWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        let StoredWeight::Sliced(slices) = table else {
            return Err(EngError::BadRequest("rows needs a sliced table".into()));
        };
        let mut out = Vec::new();
        for &row in rows {
            let packed = stored::read_slice(self.ckpt, slices, row)?;
            out.extend(View::new(&packed)?.dequantize_row(0));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::fixtures::{self, bf16};
    use crate::kernels::QuantizationSpec;
    use crate::tensor::Tensor;

    /// Every word holds the nibbles 0 to 7 from the low end, so column `c` has code `c % 8`.
    const NIBBLES: u32 = 0x7654_3210;

    fn spec(bits: u32) -> QuantizationSpec {
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits,
            group_size: 64,
        }
    }

    fn bf16_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
        let bytes = values.iter().flat_map(|v| bf16(*v)).collect();
        Tensor::try_from_host(Dtype::Bf16, shape, bytes).unwrap()
    }

    /// A packed matrix `[rows, cols]` of 4-bit codes `c % 8`, with one scale and bias per group.
    fn nibble_weight(rows: u32, cols: u32, scales: &[f32], biases: &[f32]) -> PackedWeight {
        let words = (rows * cols / 8) as usize;
        let payload = Tensor::try_from_host(
            Dtype::U32,
            vec![rows, cols / 8],
            NIBBLES.to_le_bytes().repeat(words),
        )
        .unwrap();
        let shape = vec![rows, cols / 64];
        PackedWeight::try_new(
            payload,
            bf16_tensor(shape.clone(), scales),
            Some(bf16_tensor(shape, biases)),
            vec![rows, cols],
            spec(4),
        )
        .unwrap()
    }

    fn expect_close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (f64::from(*a) - e).abs() <= 1e-4 * (1.0 + e.abs()),
                "[{i}] {a} vs {e}"
            );
        }
    }

    #[test]
    fn four_bit_rows_use_the_scale_and_bias_of_their_group() {
        // Two rows, two groups per row. The value at column c of group g in row r is
        // scale[r][g] * (c % 8) + bias[r][g].
        let scales = [1.0, 0.5, 2.0, 4.0];
        let biases = [0.0, 1.0, 0.5, -1.0];
        let weight = nibble_weight(2, 128, &scales, &biases);
        let x: Vec<f32> = (0..128).map(|c| (c % 5) as f32 - 1.0).collect();
        let expected: Vec<f64> = (0..2)
            .map(|r| {
                (0..128)
                    .map(|c| {
                        let g = r * 2 + c / 64;
                        f64::from(x[c])
                            * (f64::from(scales[g]) * (c % 8) as f64 + f64::from(biases[g]))
                    })
                    .sum()
            })
            .collect();
        expect_close(&matvec(&weight, &x, 1).unwrap(), &expected);
        assert_eq!(
            matvec(&weight, &x, 1).unwrap(),
            matvec(&weight, &x, 3).unwrap()
        );
    }

    #[test]
    fn eight_bit_rows_read_one_byte_per_column() {
        // The word bytes 0x00, 0x03, 0x07, 0xFF repeat, so column c has code [0, 3, 7, 255][c % 4].
        // With x = 1 the code sum is 16 * 265 = 4240, so the dot is 1.0078125 * 4240 + 0.5 * 64.
        let payload = Tensor::try_from_host(
            Dtype::U32,
            vec![1, 16],
            0xFF07_0300u32.to_le_bytes().repeat(16),
        )
        .unwrap();
        let weight = PackedWeight::try_new(
            payload,
            bf16_tensor(vec![1, 1], &[1.007_812_5]),
            Some(bf16_tensor(vec![1, 1], &[0.5])),
            vec![1, 64],
            spec(8),
        )
        .unwrap();
        assert_eq!(matvec(&weight, &[1.0; 64], 1).unwrap(), [4305.125]);
    }

    #[test]
    fn two_bit_rows_read_sixteen_codes_per_word_from_the_low_end() {
        // Row 0 words 0xE4E4E4E4 hold the codes 0, 1, 2, 3 from the low end (0xE4 = 0b11_10_01_00).
        // Row 1 words 0x1B1B1B1B hold 3, 2, 1, 0. With x = 1 at c % 4 == 3 and 0 elsewhere, the
        // sixteen picked columns have code 3 in row 0 and code 0 in row 1:
        // row 0 = 0.5 * 3 * 16 + 1.0 * 16 = 40, row 1 = 2.0 * 0 * 16 + (-1.0) * 16 = -16.
        let words: Vec<u8> = [0xE4E4_E4E4u32, 0x1B1B_1B1B]
            .iter()
            .flat_map(|w| w.to_le_bytes().repeat(4))
            .collect();
        let payload = Tensor::try_from_host(Dtype::U32, vec![2, 4], words).unwrap();
        let weight = PackedWeight::try_new(
            payload,
            bf16_tensor(vec![2, 1], &[0.5, 2.0]),
            Some(bf16_tensor(vec![2, 1], &[1.0, -1.0])),
            vec![2, 64],
            spec(2),
        )
        .unwrap();
        let x: Vec<f32> = (0..64).map(|c| f32::from(c % 4 == 3)).collect();
        let out = matvec(&weight, &x, 1);
        assert!(out.is_ok(), "2-bit weights are supported");
        assert_eq!(out.unwrap(), [40.0, -16.0]);
    }

    #[test]
    fn f32_scales_and_biases_are_read_four_bytes_per_group() {
        // The scales 1 + 2^-10 and 2 + 2^-9 are not BF16 values (BF16 keeps 7 mantissa bits).
        // The codes are c % 8, so a group of 64 columns has code sum 8 * 28 = 224. With x = 1, group 0
        // gives (1 + 2^-10) * 224 + 0.5 * 64 = 256.21875 and group 1 gives (2 + 2^-9) * 224 - 0.25 * 64
        // = 432.4375, so the dot is 688.65625.
        let payload =
            Tensor::try_from_host(Dtype::U32, vec![1, 16], NIBBLES.to_le_bytes().repeat(16))
                .unwrap();
        let f32_tensor = |values: &[f32]| {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Tensor::try_from_host(Dtype::F32, vec![1, 2], bytes).unwrap()
        };
        let weight = PackedWeight::try_new(
            payload,
            f32_tensor(&[1.0 + 1.0 / 1024.0, 2.0 + 1.0 / 512.0]),
            Some(f32_tensor(&[0.5, -0.25])),
            vec![1, 128],
            spec(4),
        )
        .unwrap();
        assert_eq!(matvec(&weight, &[1.0; 128], 1).unwrap(), [688.65625]);
    }

    #[test]
    fn unsupported_weights_and_wrong_inputs_are_rejected() {
        let weight = nibble_weight(1, 64, &[1.0], &[0.0]);
        assert!(matches!(
            matvec(&weight, &[1.0; 63], 1),
            Err(EngError::BadRequest(_))
        ));
        // Three-bit codes do not pack into whole bytes per column here.
        let payload = Tensor::try_from_host(Dtype::U32, vec![1, 6], vec![0; 24]).unwrap();
        let three = PackedWeight::try_new(
            payload,
            bf16_tensor(vec![1, 1], &[1.0]),
            Some(bf16_tensor(vec![1, 1], &[0.0])),
            vec![1, 64],
            spec(3),
        )
        .unwrap();
        assert!(matches!(
            matvec(&three, &[1.0; 64], 1),
            Err(EngError::BadRequest(_))
        ));
    }

    /// A checkpoint with a 3-expert stack `[3, 2, 64]` and a 4-row table `[4, 64]`, 4-bit codes `c % 8`.
    /// Expert `e` has scale `e + 1` and bias `0.25 * row`. Table row `r` has scale `r + 1` and bias `0.5`.
    fn slices_checkpoint() -> StreamedCheckpoint {
        let mut entries = Vec::new();
        fixtures::packed(
            &mut entries,
            ("m.stack", &[3, 2], 64),
            (4, 64),
            |_| NIBBLES,
            |g| (g / 2 + 1) as f32,
            |g| 0.25 * (g % 2) as f32,
        );
        fixtures::packed(
            &mut entries,
            ("m.table", &[4], 64),
            (4, 64),
            |_| NIBBLES,
            |g| (g + 1) as f32,
            |_| 0.5,
        );
        fixtures::checkpoint(entries)
    }

    #[test]
    fn expert_slices_and_table_rows_come_from_the_checkpoint() {
        let ckpt = slices_checkpoint();
        let stack = StoredWeight::Sliced(stored::sliced(&ckpt, "m.stack", spec(4), 2, 64).unwrap());
        let table = StoredWeight::Sliced(stored::sliced(&ckpt, "m.table", spec(4), 1, 64).unwrap());
        let mut projector = CpuProjector::new(&ckpt, 2);
        // Expert 2 has scale 3. With x = 1 the code sum is 224, so row 0 is 3 * 224 and row 1 adds 0.25 * 64.
        expect_close(
            &projector.expert_linear(&stack, 2, &[1.0; 64]).unwrap(),
            &[672.0, 688.0],
        );
        assert!(matches!(
            projector.expert_linear(&stack, 3, &[1.0; 64]),
            Err(EngError::BadRequest(_))
        ));
        // Rows 3 and 1 come back in request order: scale * (c % 8) + 0.5 with scale 4 and 2.
        let rows = projector.rows(&table, &[3, 1]).unwrap();
        let expected: Vec<f64> = [4.0, 2.0]
            .iter()
            .flat_map(|scale| (0..64).map(move |c| scale * (c % 8) as f64 + 0.5))
            .collect();
        expect_close(&rows, &expected);
    }

    #[test]
    fn linear_and_expert_linear_reject_the_wrong_weight_kind() {
        let ckpt = slices_checkpoint();
        let stack = StoredWeight::Sliced(stored::sliced(&ckpt, "m.stack", spec(4), 2, 64).unwrap());
        let packed = StoredWeight::Packed(Box::new(nibble_weight(1, 64, &[1.0], &[0.0])));
        let mut projector = CpuProjector::new(&ckpt, 1);
        assert!(matches!(
            projector.linear(&stack, &[1.0; 64]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            projector.expert_linear(&packed, 0, &[1.0; 64]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            projector.rows(&packed, &[0]),
            Err(EngError::BadRequest(_))
        ));
    }
}
