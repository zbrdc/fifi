use super::{E4M3, Nvfp4Matrix, RowResult, group_offsets};
use std::{arch::aarch64::*, ops::Range};

// Integers representing twice the E2M1 value. Conversion below shifts by one bit.
const E2M1_TWICE: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[target_feature(enable = "neon")]
fn values(table: int8x16_t, indices: uint8x8_t) -> [float32x4_t; 2] {
    let values = vmovl_s8(vqtbl1_s8(table, indices));
    [
        vcvtq_n_f32_s32::<1>(vmovl_s16(vget_low_s16(values))),
        vcvtq_n_f32_s32::<1>(vmovl_s16(vget_high_s16(values))),
    ]
}

/// Decode one group of eight packed bytes for each set of up to four activation rows.
#[target_feature(enable = "neon")]
pub(super) fn dot_rows<const NX: usize>(
    m: &Nvfp4Matrix<'_>,
    rows: Range<usize>,
    xs: [&[f32]; NX],
    res: &mut [RowResult],
    max_code: &mut u8,
) {
    // SAFETY: this table holds 16 consecutive i8 entries.
    let table = unsafe { vld1q_s8(E2M1_TWICE.as_ptr()) };
    let nibble = vdup_n_u8(15);
    for (out, row) in res.iter_mut().zip(rows) {
        let scales = m.row_scales(row);
        let groups = m.row_packed(row).as_chunks::<8>().0;
        let mut accum = [vdupq_n_f32(0.0); NX];
        for (group, (&code, bytes)) in scales.iter().zip(groups).enumerate() {
            *max_code = (*max_code).max(code);
            let scale = E4M3[usize::from(code)];
            // SAFETY: a group has eight consecutive packed bytes.
            let packed = unsafe { vld1_u8(bytes.as_ptr()) };
            let [we0, we1] = values(table, vand_u8(packed, nibble));
            let [wo0, wo1] = values(table, vshr_n_u8::<4>(packed));
            let (even, odd) = group_offsets(m.cols, group);
            for (acc, x) in accum.iter_mut().zip(xs) {
                // SAFETY: group_offsets keeps both halves inside an input of m.cols elements.
                let [xe0, xe1, xo0, xo1] = unsafe {
                    [
                        vld1q_f32(x.as_ptr().add(even)),
                        vld1q_f32(x.as_ptr().add(even + 4)),
                        vld1q_f32(x.as_ptr().add(odd)),
                        vld1q_f32(x.as_ptr().add(odd + 4)),
                    ]
                };
                let dot = vfmaq_f32(vmulq_f32(we0, xe0), wo0, xo0);
                let dot = vfmaq_f32(dot, we1, xe1);
                let dot = vfmaq_f32(dot, wo1, xo1);
                *acc = vfmaq_n_f32(*acc, dot, scale);
            }
        }
        for (slot, acc) in out.iter_mut().zip(accum) {
            *slot = vaddvq_f32(acc) * m.weight_scale_2;
        }
    }
}
