//! An f32 exponential that every platform and the CUDA kernels compute bit for bit.

/// 2^r = Σ (r ln 2)^i / i! for |r| <= 1/2, to the ninth power, as double bits.
const EXP2_TERMS: [u64; 10] = [
    0x3ff0_0000_0000_0000,
    0x3fe6_2e42_fefa_39ef,
    0x3fce_bfbd_ff82_c58f,
    0x3fac_6b08_d704_a0c0,
    0x3f83_b2ab_6fba_4e77,
    0x3f55_d87f_e78a_6731,
    0x3f24_3091_2f86_c787,
    0x3eef_fcbf_c588_b0c7,
    0x3eb6_2c02_23a5_c824,
    0x3e7b_5253_d395_e7c4,
];
/// 1 / ln 2 as double bits.
const INV_LN2: u64 = 0x3ff7_1547_652b_82fe;

/// e^x from IEEE double additions and multiplications only, so every host and the CUDA kernels
/// that share it (`portable_exp` in depth_attention.cu) compute the same bits, whatever the
/// platform's libm. Within one float ulp of e^x.
pub(crate) fn portable_exp(x: f32) -> f32 {
    // Adding and removing 1.5 * 2^52 rounds a double of magnitude below 2^51 to an integer.
    const SHIFT: f64 = 6_755_399_441_055_744.0;
    if x.is_nan() {
        return x;
    }
    if x <= -110.0 {
        return 0.0;
    }
    if x > 89.0 {
        return f32::INFINITY;
    }
    // e^x = 2^z = 2^k * 2^r with integer k and |r| <= 1/2.
    let z = f64::from(x) * f64::from_bits(INV_LN2);
    let k = (z + SHIFT) - SHIFT;
    let r = z - k;
    let p = EXP2_TERMS[..9]
        .iter()
        .rev()
        .fold(f64::from_bits(EXP2_TERMS[9]), |p, &term| {
            p * r + f64::from_bits(term)
        });
    // k lies in [-159, 129], so 2^k is a normal double and the product is exact before rounding.
    let scale = f64::from_bits(((k as i64 + 1023) as u64) << 52);
    (p * scale) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_within_one_float_ulp_of_the_double_exponential() {
        let ordered = |x: f32| {
            let bits = x.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        let mut state = 99_u64;
        for _ in 0..1_000_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = -((state >> 40) as f32) * (110.0 / (1 << 24) as f32);
            let (ours, wide) = (portable_exp(x), f64::from(x).exp() as f32);
            assert!(
                (ordered(ours) - ordered(wide)).abs() <= 1,
                "exp({x:e}): {ours:e} against {wide:e}"
            );
        }
        assert_eq!(portable_exp(0.0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(portable_exp(-0.0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(portable_exp(f32::NEG_INFINITY), 0.0);
        assert!(portable_exp(f32::NAN).is_nan());
        assert_eq!(portable_exp(100.0), f32::INFINITY);
    }
}
