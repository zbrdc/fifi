//! The request side of the canonical device sampler.
//!
//! The CUDA kernels (not yet landed) and the Rust reference in
//! `reference_tests` are to implement ONE definition of a draw and agree bit for bit. This
//! file holds the part the executor needs at run time: the exact encoding of a request's
//! `temperature` and `top_p` that the start block carries to the device.
//!
//! - The device multiplies by `inv_t`, so the host rounds `1/T` once, in f64.
//! - The nucleus threshold is `ceil(top_p_m * total / 2^top_p_e)` over exact integers. It needs
//!   `top_p` as an exact dyadic fraction, because `top_p * total` in f64 would round.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the CUDA sampler (SampleParams::draw)")
)]

use crate::EngError;

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
pub(crate) mod reference_tests;
#[cfg(test)]
mod tests;

/// The contract maximum for `temperature` (`SamplingParams::new` checks the same bound).
const MAX_TEMPERATURE: f64 = 2.0;
/// The exponent of a subnormal `top_p`: the value is `fraction * 2^-1074`.
const SUBNORMAL_EXPONENT: i32 = 1074;
/// The offset from a normal biased exponent to `top_p_e` (52 fraction bits, bias 1023).
const NORMAL_EXPONENT_OFFSET: i32 = 1075;
const FRACTION_BITS: u32 = 52;
const FRACTION_MASK: u64 = (1 << FRACTION_BITS) - 1;
const IMPLICIT_ONE: u64 = 1 << FRACTION_BITS;

/// The per-request constants of one draw.
///
/// `top_p` equals `top_p_m * 2^-top_p_e` exactly. `top_p == 0` is `(0, 0)`, and `1.0` is
/// `(2^52, 52)`. A nonzero `top_p` has `top_p_e` in `52..=1074`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CanonParams {
    pub(crate) inv_t: f64,
    pub(crate) top_p_m: u64,
    pub(crate) top_p_e: i32,
}

/// Encodes a non-greedy request for the device sampler.
///
/// Refuses `temperature` outside `(0, 2]`, because greedy requests never reach the sampler,
/// and `top_p` outside `[0, 1]`. A NaN fails both checks. A `temperature` so small that its
/// reciprocal overflows gives `inv_t == +inf`, which the canonical weight rule handles.
pub(crate) fn derive(temperature: f64, top_p: f64) -> Result<CanonParams, EngError> {
    if temperature == 0.0 || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
        return Err(EngError::BadRequest(format!(
            "temperature must be above 0 and at most 2 for a sampled draw, got {temperature}"
        )));
    }
    if !(0.0..=1.0).contains(&top_p) {
        return Err(EngError::BadRequest(format!(
            "top_p must be between 0 and 1, got {top_p}"
        )));
    }
    let (top_p_m, top_p_e) = split_top_p(top_p);
    Ok(CanonParams {
        inv_t: 1.0 / temperature,
        top_p_m,
        top_p_e,
    })
}

/// Splits `top_p` in `[0, 1]` into `(m, e)` with `top_p == m * 2^-e` exactly.
fn split_top_p(top_p: f64) -> (u64, i32) {
    if top_p == 0.0 {
        return (0, 0);
    }
    let bits = top_p.to_bits();
    let biased = (bits >> FRACTION_BITS) & 0x7FF;
    let fraction = bits & FRACTION_MASK;
    if biased == 0 {
        return (fraction, SUBNORMAL_EXPONENT);
    }
    (
        fraction | IMPLICIT_ONE,
        NORMAL_EXPONENT_OFFSET - biased as i32,
    )
}
