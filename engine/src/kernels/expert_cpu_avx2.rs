#![allow(unsafe_code)]
//! AVX2 integer row dots for Q4_K weights against Q8_K rows and against per-32 group rows
//! (the CPU expert tier dispatches to it through `DotBackend`).
//!
//! Scope: the repo's second scoped intrinsics module, confined to this file by the
//! `unsafe_is_confined_to_the_avx2_module` test; the first, the tiered mapped storage,
//! set the scoped-module pattern. What lives here: two AVX2 row dots
//! (`q4k_q8k_row_dot_avx2` for the Q8_K grid and `q4k_q8g_row_dot_avx2` for
//! the 32-value group grid of the device path) plus their safe entry wrappers; the safe
//! scalar paths in `expert_cpu_q8k` and `expert_cpu_q8g` stay the references and the
//! fallbacks, and the f32 `expert_forward` remains the overall reference.
//!
//! Numerics: the SIMD part computes only EXACT integers. Per 256-value block, 8 i32
//! lanes accumulate the scale-weighted partial sums of the 8 sub-blocks: nibbles are
//! split with `and`/`srli` exactly as the scalar path indexes them, each 32-value
//! sub-block is dotted with `maddubs` (FIRST operand the unsigned nibbles, SECOND the
//! signed Q8 bytes; one i16 pair is at most `2 * 15 * 127 = 3810`, inside i16),
//! widened exactly by `madd` against ones, scaled by the 6-bit sub-block scale with
//! `mullo_epi32` and added (worst lane `8 * 63 * 4 * 15 * 127 = 3.8M`, far inside
//! i32); the horizontal lane sum is the exact `sumi`. `summ` and the float tail are
//! the shared scalar code (`q8k_block_value`), so the scalar and AVX2 paths are
//! bitwise identical.
//!
//! The per-32 dot (see `expert_cpu_q8g` for the numerics) computes, per 256-value block, the
//! eight sub-block sums `isum_j` exactly (`maddubs`/`madd` per sub-block, then a `hadd` tree
//! over the eight vectors), scales them with `mullo_epi32`, converts to f32 (exact, below
//! 2^24), multiplies by the group scales (one rounding, as the scalar), and adds the products
//! in the fixed tree order of `tree8` with two `hadd_ps` and one lane add, so the result
//! is bitwise the scalar one.
//!
//! With the narrow AVX2 path opted in, F16C can convert both Q4_K scales together.
//! CPUs without F16C retain the scalar conversion.
//!
//! Residual obligations: at most 8 unsafe blocks (including the F16C conversion test),
//! each with a SAFETY proof comment
//! naming the runtime feature check, the caller's length validation and the in-bounds
//! unaligned loads; on non-x86_64 targets this module compiles to an empty,
//! documented stub, where `DotBackend::detect` never reports `Avx2` and the
//! dispatching matvec falls back to the scalar reference.

#[cfg(target_arch = "x86_64")]
pub(crate) use imp::{q4k_q8g_row_dot, q4k_q8k_row_dot};

#[cfg(target_arch = "x86_64")]
mod imp {
    use crate::kernels::expert_cpu::{
        Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES, f16_bits_to_f32, scale_min,
    };
    use crate::kernels::expert_cpu_q8g::{
        GROUPS_PER_BLOCK, Q8gRow, q4k_q8g_row_dot_scalar, q8g_block_value,
    };
    use crate::kernels::expert_cpu_q8k::{Q8kRow, q4k_q8k_row_dot_scalar, q8k_block_value};
    use std::arch::x86_64::*;

    /// Safe entry to the AVX2 dot: re-checks the runtime feature bit (so no caller
    /// mistake can run the intrinsics on a CPU without AVX2) and falls back to the
    /// scalar reference when it is missing. Slice lengths are the caller's contract:
    /// the dispatching matvec in `expert_cpu_q8k` validates `row` against `xq` first.
    pub(crate) fn q4k_q8k_row_dot(row: &[u8], xq: &Q8kRow) -> f32 {
        if !std::is_x86_feature_detected!("avx2") {
            return q4k_q8k_row_dot_scalar(row, xq);
        }
        // SAFETY: the AVX2 feature bit was detected at runtime just above, so the
        // SAFETY: target-feature function may run on this CPU; slice lengths were
        // SAFETY: validated by the dispatching matvec before this call, and every
        // SAFETY: load inside is an unaligned `loadu` within those validated bounds.
        unsafe { q4k_q8k_row_dot_avx2(row, xq) }
    }

    /// AVX2 row dot of `n` whole Q4_K blocks with a quantized row: per block, the 8
    /// i32 lanes hold the scale-weighted partial sums whose exact horizontal sum is
    /// `sumi`; `summ` and the float tail run in the shared scalar code, so the result
    /// is bitwise identical to `q4k_q8k_row_dot_scalar`.
    #[target_feature(enable = "avx2")]
    unsafe fn q4k_q8k_row_dot_avx2(row: &[u8], xq: &Q8kRow) -> f32 {
        let mut acc = 0.0f32;
        for (b, blk) in row.as_chunks::<Q4K_BLOCK_BYTES>().0.iter().enumerate() {
            let d_w = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin_w = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let q8 = &xq.q[Q4K_BLOCK_VALUES * b..][..Q4K_BLOCK_VALUES];
            let bsums = &xq.bsums[16 * b..][..16];
            // SAFETY: the safe wrapper above checked the runtime AVX2 feature bit
            // SAFETY: before entry; every load reads 32 bytes through an unaligned
            // SAFETY: `loadu` from slices already bounds-checked above (`blk` is a
            // SAFETY: whole 144-byte block, `q8` a whole 256-value block), so all
            // SAFETY: reads stay in bounds with no alignment assumption.
            let sumi = unsafe {
                let nib = _mm256_set1_epi8(0x0f);
                let ones = _mm256_set1_epi16(1);
                let mut lanes = _mm256_setzero_si256();
                let (wchunks, _) = blk[16..].as_chunks::<32>();
                let (qchunks, _) = q8.as_chunks::<64>();
                for (c, (wbytes, qc)) in wchunks.iter().zip(qchunks).enumerate() {
                    let w = _mm256_loadu_si256(wbytes.as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(w, nib);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), nib);
                    let qa = _mm256_loadu_si256(qc.as_ptr() as *const __m256i);
                    let qb = _mm256_loadu_si256(qc[32..].as_ptr() as *const __m256i);
                    let pa = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, qa), ones);
                    let pb = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, qb), ones);
                    let (sc_a, _) = scale_min(2 * c, blk);
                    let (sc_b, _) = scale_min(2 * c + 1, blk);
                    lanes = _mm256_add_epi32(
                        lanes,
                        _mm256_mullo_epi32(pa, _mm256_set1_epi32(i32::from(sc_a))),
                    );
                    lanes = _mm256_add_epi32(
                        lanes,
                        _mm256_mullo_epi32(pb, _mm256_set1_epi32(i32::from(sc_b))),
                    );
                }
                // Exact horizontal sum of the 8 i32 lanes (integer adds commute).
                let lo128 = _mm256_castsi256_si128(lanes);
                let hi128 = _mm256_extracti128_si256(lanes, 1);
                let s = _mm_add_epi32(lo128, hi128);
                let s2 = _mm_add_epi32(s, _mm_srli_si128(s, 8));
                let s3 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
                _mm_cvtsi128_si32(s3)
            };
            let mut summ = 0i32;
            for j in 0..8 {
                let (_, m) = scale_min(j, blk);
                let msum = i32::from(bsums[2 * j]) + i32::from(bsums[2 * j + 1]);
                summ += i32::from(m) * msum;
            }
            acc += q8k_block_value(xq.d[b], d_w, dmin_w, sumi, summ);
        }
        acc
    }

    /// The narrow path follows upstream's default-on setting. F16C is
    /// separately opt-in so the existing default AVX2 path stays available.
    fn q4k_q8g_row_dot_dispatch(row: &[u8], xq: &Q8gRow, forced: Option<bool>) -> f32 {
        static NARROW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static F16C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let narrow = forced.unwrap_or_else(|| {
            *NARROW
                .get_or_init(|| crate::env_switch::env_allows_default_on("FIFI_CPU_Q8G_NARROW_SUM"))
        });
        let f16c = forced.is_none()
            && narrow
            && *F16C.get_or_init(|| {
                std::env::var_os("FIFI_CPU_Q8G_F16C_SCALES").is_some_and(|value| value == "1")
                    && std::is_x86_feature_detected!("f16c")
            });
        // SAFETY: the caller checked AVX2 and validated the input lengths.
        // SAFETY: the F16C entry also requires its detected feature bit.
        unsafe {
            if f16c {
                q4k_q8g_row_dot_f16c(row, xq)
            } else if narrow {
                q4k_q8g_row_dot_avx2::<true>(row, xq)
            } else {
                q4k_q8g_row_dot_avx2::<false>(row, xq)
            }
        }
    }

    /// Safe entry to the per-32 AVX2 dot with runtime fallback to scalar.
    pub(crate) fn q4k_q8g_row_dot(row: &[u8], xq: &Q8gRow) -> f32 {
        if !std::is_x86_feature_detected!("avx2") {
            return q4k_q8g_row_dot_scalar(row, xq);
        }
        q4k_q8g_row_dot_dispatch(row, xq, None)
    }

    /// Test-only forced-path entry for the narrow versus wide equality sweep.
    #[cfg(test)]
    pub(crate) fn q4k_q8g_row_dot_forced<const NARROW: bool>(row: &[u8], xq: &Q8gRow) -> f32 {
        if !std::is_x86_feature_detected!("avx2") {
            return q4k_q8g_row_dot_scalar(row, xq);
        }
        q4k_q8g_row_dot_dispatch(row, xq, Some(NARROW))
    }

    #[inline]
    #[target_feature(enable = "f16c")]
    unsafe fn block_scales_f16c(blk: &[u8; Q4K_BLOCK_BYTES]) -> (f32, f32) {
        let halves = _mm_cvtsi32_si128(i32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]));
        let scales = _mm_cvtph_ps(halves);
        (
            _mm_cvtss_f32(scales),
            _mm_cvtss_f32(_mm_shuffle_ps(scales, scales, 1)),
        )
    }

    /// The `isum_j` of the eight sub-blocks of one block as eight exact i32 lanes: per sub-block
    /// reduce product pairs at either i16 or i32 width, then combine the two 128-bit halves.
    /// Both trees retain the original group order and compute exact integer sums.
    #[target_feature(enable = "avx2")]
    unsafe fn sub_block_sums<const NARROW: bool>(blk: &[u8], q8: &[i8]) -> __m256i {
        // SAFETY: called only from `q4k_q8g_row_dot_avx2` after the runtime AVX2 check; `blk`
        // SAFETY: is a whole 144-byte block and `q8` a whole 256-value block, so every 32-byte
        // SAFETY: unaligned load below stays inside them.
        unsafe {
            let nib = _mm256_set1_epi8(0x0f);
            let ones = _mm256_set1_epi16(1);
            let mut p = [_mm256_setzero_si256(); GROUPS_PER_BLOCK];
            let (wchunks, _) = blk[16..].as_chunks::<32>();
            let (qchunks, _) = q8.as_chunks::<64>();
            for (c, (wbytes, qc)) in wchunks.iter().zip(qchunks).enumerate() {
                let w = _mm256_loadu_si256(wbytes.as_ptr() as *const __m256i);
                let lo = _mm256_and_si256(w, nib);
                let hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), nib);
                let qa = _mm256_loadu_si256(qc.as_ptr() as *const __m256i);
                let qb = _mm256_loadu_si256(qc[32..].as_ptr() as *const __m256i);
                let a = _mm256_maddubs_epi16(lo, qa);
                let b = _mm256_maddubs_epi16(hi, qb);
                if NARROW {
                    p[2 * c] = a;
                    p[2 * c + 1] = b;
                } else {
                    p[2 * c] = _mm256_madd_epi16(a, ones);
                    p[2 * c + 1] = _mm256_madd_epi16(b, ones);
                }
            }
            let (ab, cd) = if NARROW {
                // Each i16 initially holds two products. After two horizontal adds
                // it holds eight, bounded by 8 * 15 * 128 = 15360. Widen before
                // joining the two halves of the 32-value group.
                let four_groups = |a, b, c, d| {
                    let ab = _mm256_hadd_epi16(a, b);
                    let cd = _mm256_hadd_epi16(c, d);
                    _mm256_madd_epi16(_mm256_hadd_epi16(ab, cd), ones)
                };
                (
                    four_groups(p[0], p[1], p[2], p[3]),
                    four_groups(p[4], p[5], p[6], p[7]),
                )
            } else {
                let a = _mm256_hadd_epi32(p[0], p[1]);
                let b = _mm256_hadd_epi32(p[2], p[3]);
                let c = _mm256_hadd_epi32(p[4], p[5]);
                let d = _mm256_hadd_epi32(p[6], p[7]);
                (_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d))
            };
            let lo = _mm_add_epi32(_mm256_castsi256_si128(ab), _mm256_extracti128_si256(ab, 1));
            let hi = _mm_add_epi32(_mm256_castsi256_si128(cd), _mm256_extracti128_si256(cd, 1));
            _mm256_set_m128i(hi, lo)
        }
    }

    /// AVX2 row dot of `n` whole Q4_K blocks with a per-32 group row: bitwise identical to
    /// `q4k_q8g_row_dot_scalar` (exact integers, the same f32 products, the `tree8` order).
    #[target_feature(enable = "avx2")]
    unsafe fn q4k_q8g_row_dot_avx2<const NARROW: bool>(row: &[u8], xq: &Q8gRow) -> f32 {
        // SAFETY: the caller checked AVX2 and row lengths. This entry uses scalar scales.
        unsafe { q4k_q8g_row_dot_inner::<NARROW, false>(row, xq) }
    }

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn q4k_q8g_row_dot_f16c(row: &[u8], xq: &Q8gRow) -> f32 {
        // SAFETY: the caller checked AVX2, F16C and row lengths.
        unsafe { q4k_q8g_row_dot_inner::<true, true>(row, xq) }
    }

    /// Requires AVX2 and validated row lengths, plus F16C for HARDWARE_SCALES.
    #[inline(always)]
    unsafe fn q4k_q8g_row_dot_inner<const NARROW: bool, const HARDWARE_SCALES: bool>(
        row: &[u8],
        xq: &Q8gRow,
    ) -> f32 {
        let mut acc = 0.0f32;
        for (b, blk) in row.as_chunks::<Q4K_BLOCK_BYTES>().0.iter().enumerate() {
            let q8 = &xq.q[Q4K_BLOCK_VALUES * b..][..Q4K_BLOCK_VALUES];
            let dx = &xq.d[GROUPS_PER_BLOCK * b..][..GROUPS_PER_BLOCK];
            let sums = &xq.s[GROUPS_PER_BLOCK * b..][..GROUPS_PER_BLOCK];
            let (mut sc, mut mn) = ([0i32; GROUPS_PER_BLOCK], [0i32; GROUPS_PER_BLOCK]);
            for j in 0..GROUPS_PER_BLOCK {
                let (s, m) = scale_min(j, blk);
                (sc[j], mn[j]) = (i32::from(s), i32::from(m));
            }
            // SAFETY: the caller checked AVX2 and, for HARDWARE_SCALES, F16C.
            // SAFETY: `dx` and `sums` are exactly 8 elements (32 bytes), so the two unaligned
            // SAFETY: 32-byte loads read inside them; the other calls take values or the
            // SAFETY: bounds-checked slices of `sub_block_sums`.
            let (d_w, dmin_w, t1, t2) = unsafe {
                let (d_w, dmin_w) = if HARDWARE_SCALES {
                    block_scales_f16c(blk)
                } else {
                    (
                        f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
                        f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
                    )
                };
                let isum = sub_block_sums::<NARROW>(blk, q8);
                let sc_v =
                    _mm256_setr_epi32(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5], sc[6], sc[7]);
                let mn_v =
                    _mm256_setr_epi32(mn[0], mn[1], mn[2], mn[3], mn[4], mn[5], mn[6], mn[7]);
                let dxv = _mm256_loadu_ps(dx.as_ptr());
                let sv = _mm256_loadu_si256(sums.as_ptr() as *const __m256i);
                // Exact integers below 2^24, converted exactly, one rounding in the multiply.
                let p1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(isum, sc_v)), dxv);
                let p2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(sv, mn_v)), dxv);
                // tree8 for both products at once: hadd pairs, hadd pairs of pairs, then the
                // low half plus the high half: [(p0+p1)+(p2+p3)] + [(p4+p5)+(p6+p7)].
                let h = _mm256_hadd_ps(p1, p2);
                let h2 = _mm256_hadd_ps(h, h);
                let t = _mm_add_ps(_mm256_castps256_ps128(h2), _mm256_extractf128_ps(h2, 1));
                (
                    d_w,
                    dmin_w,
                    _mm_cvtss_f32(t),
                    _mm_cvtss_f32(_mm_shuffle_ps(t, t, 1)),
                )
            };
            acc += q8g_block_value(d_w, dmin_w, t1, t2);
        }
        acc
    }

    #[test]
    fn f16c_weight_scales_match_every_binary16_value() {
        if !std::is_x86_feature_detected!("f16c") {
            return;
        }
        let mut block = [0u8; Q4K_BLOCK_BYTES];
        for bits in 0..=u16::MAX {
            let minimum = bits.reverse_bits();
            block[..2].copy_from_slice(&bits.to_le_bytes());
            block[2..4].copy_from_slice(&minimum.to_le_bytes());
            // SAFETY: the test detected F16C and supplies one complete Q4_K block.
            let (d, dmin) = unsafe { block_scales_f16c(&block) };
            assert_eq!(d.to_bits(), f16_bits_to_f32(bits).to_bits());
            assert_eq!(dmin.to_bits(), f16_bits_to_f32(minimum).to_bits());
        }
    }
}

/// Test-only re-export of the forced-path dot so the q8g sweep (a sibling
/// module's tests) can force both paths.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use imp::q4k_q8g_row_dot_forced;
