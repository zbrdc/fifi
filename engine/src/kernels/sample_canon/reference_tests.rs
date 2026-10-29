//! The Rust reference of the canonical device sampler. Test-only.
//!
//! The CUDA kernels must reproduce `reference_sample` bit for bit (parity level P1). It is
//! deliberately naive: a full sort, then sequential sums over exact integers. The device
//! reaches the same numbers with histograms, because integer addition is associative.
//!
//! Every f64 step is IEEE, and each fused multiply-add is written with `mul_add`, so a
//! compiler cannot contract or reorder it.

use super::CanonParams;

/// The largest row the canonical sampler accepts. It keeps every mass below 2^63.
pub(crate) const MAX_N: usize = 1 << 18;
/// `CanonDraw::path` when the draw needs no randomness: the token is the argmax.
pub(crate) const PATH_ARGMAX: i32 = 1;
/// `CanonDraw::path` when a unit picked the token.
pub(crate) const PATH_DRAW: i32 = 2;

const PHILOX_ROUNDS: u32 = 10;
const PHILOX_MUL0: u32 = 0xD251_1F53;
const PHILOX_MUL1: u32 = 0xCD9E_8D57;
const PHILOX_KEY_BUMP0: u32 = 0x9E37_79B9;
const PHILOX_KEY_BUMP1: u32 = 0xBB67_AE85;
/// The unit holds 53 bits: the high bits of the first two Philox output words.
const UNIT_BITS: u32 = 53;

const LOG2_E: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
/// `FACTORIALS[i]` is `i!`, exactly.
const FACTORIALS: [f64; 14] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
];
/// Weights are fixed point with 45 fractional bits, so the top token weighs exactly 2^45.
const WEIGHT_SCALE: f64 = (1_u64 << 45) as f64;
/// A weight below this exponent argument rounds to 0 in fixed point.
const WEIGHT_CUTOFF: f64 = -32.0;

fn high(value: u64) -> u32 {
    (value >> 32) as u32
}

fn low(value: u64) -> u32 {
    value as u32
}

fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let product0 = u64::from(PHILOX_MUL0) * u64::from(counter[0]);
    let product1 = u64::from(PHILOX_MUL1) * u64::from(counter[2]);
    [
        high(product1) ^ counter[1] ^ key[0],
        low(product1),
        high(product0) ^ counter[3] ^ key[1],
        low(product0),
    ]
}

/// Philox4x32-10 as published in Random123: ten rounds, the key bumped between rounds.
pub(crate) fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut state = counter;
    let mut round_key = key;
    for round in 0..PHILOX_ROUNDS {
        if round > 0 {
            round_key = [
                round_key[0].wrapping_add(PHILOX_KEY_BUMP0),
                round_key[1].wrapping_add(PHILOX_KEY_BUMP1),
            ];
        }
        state = philox_round(state, round_key);
    }
    state
}

/// The 53-bit unit of the token at absolute position `p`.
///
/// The key is the seed, the counter is `(p, 0, 0)` split into 32-bit words, and the unit is the
/// top 53 bits of the first two output words. Counter word 2 is 0: the token-draw stream.
pub(crate) fn unit(seed: u64, p: u64) -> u64 {
    let output = philox4x32_10([low(p), high(p), 0, 0], [low(seed), high(seed)]);
    ((u64::from(output[1]) << 32) | u64::from(output[0])) >> 11
}

/// `e^x` for `x` in `[-32, 0]`, built only from IEEE operations so the device matches it.
pub(crate) fn exp_canon(x: f64) -> f64 {
    let k = (x * LOG2_E).round_ties_even();
    let reduced = (-k).mul_add(LN2_HI, x);
    let reduced = (-k).mul_add(LN2_LO, reduced);
    let mut poly = 1.0 / FACTORIALS[13];
    for factorial in FACTORIALS[..13].iter().rev() {
        poly = poly.mul_add(reduced, 1.0 / factorial);
    }
    poly * f64::from_bits(((1023 + k as i64) as u64) << 52)
}

/// The fixed-point weight of the scaled gap `x = (logit - top) / T`, which is at most 0.
pub(crate) fn weight_q(x: f64) -> u64 {
    if x < WEIGHT_CUTOFF {
        return 0;
    }
    (exp_canon(x) * WEIGHT_SCALE).round_ties_even() as u64
}

/// `floor(p / 2^s)`; 0 once the shift reaches the width.
pub(crate) fn floor_shr(p: u128, s: u32) -> u128 {
    if s >= 128 { 0 } else { p >> s }
}

/// `ceil(p / 2^s)`; 1 for a nonzero `p` once the shift reaches the width.
pub(crate) fn ceil_shr(p: u128, s: u32) -> u128 {
    let floor = floor_shr(p, s);
    let exact = if s >= 128 { p == 0 } else { floor << s == p };
    floor + u128::from(!exact)
}

/// A key that orders like the logit: -0.0 and 0.0 are equal, as in the host sampler.
pub(crate) fn value_key(v: f32) -> u32 {
    let canonical = if v == 0.0 { 0.0 } else { v };
    let bits = canonical.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The rank key of one logit: larger first, and a tie goes to the lower index.
pub(crate) fn rank_key(v: f32, index: u32) -> u64 {
    (u64::from(value_key(v)) << 32) | u64::from(!index)
}

fn index_of(key: u64) -> u32 {
    !low(key)
}

/// What the device reports for one draw (the report fields of Design 3).
///
/// On `PATH_ARGMAX` the device skips its draw kernels, so everything after `k_eff` is 0. When
/// `k_eff` equals the row length the top-k select is off, and `topk_rank_key` is 0, which keeps
/// every element as a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CanonDraw {
    pub(crate) path: i32,
    pub(crate) token: u32,
    pub(crate) keep: u32,
    pub(crate) k_eff: u32,
    pub(crate) total_q: u64,
    pub(crate) thr_q: u64,
    pub(crate) mass_q: u64,
    pub(crate) target_q: u64,
    pub(crate) topk_rank_key: u64,
}

fn narrow(value: u128) -> u64 {
    u64::try_from(value).expect("the value fits 64 bits")
}

fn effective_k(top_k: u32, n: usize) -> usize {
    let k = top_k as usize;
    if k == 0 || k >= n { n } else { k }
}

/// The scaled gap `(logit - top) / T`. A gap of exactly 0 stays 0, so `0 * inf` never occurs.
fn scaled_gap(logit: f32, top: f64, inv_t: f64) -> f64 {
    let gap = f64::from(logit) - top;
    if gap == 0.0 { 0.0 } else { gap * inv_t }
}

/// The smallest prefix length `t >= 1` whose sum reaches `theta`, and that sum.
fn select(weights: &[u64], theta: u64) -> Option<(usize, u64)> {
    let mut sum = 0_u64;
    for (i, weight) in weights.iter().enumerate() {
        sum += weight;
        if sum >= theta {
            return Some((i + 1, sum));
        }
    }
    None
}

/// The canonical draw of one row by full sort and sequential exact sums.
///
/// `unit` is the 53-bit unit; `unit(seed, p)` makes it, and the device tests inject it.
pub(crate) fn reference_sample(
    logits: &[f32],
    params: &CanonParams,
    top_k: u32,
    unit: u64,
) -> CanonDraw {
    assert!((1..=MAX_N).contains(&logits.len()), "n is outside 1..=2^18");
    assert!(logits.iter().all(|v| v.is_finite()), "logits are finite");
    assert!(unit < 1 << UNIT_BITS, "the unit holds 53 bits");
    let mut ranked: Vec<u64> = (0..)
        .zip(logits)
        .map(|(index, &logit)| rank_key(logit, index))
        .collect();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    let k_eff = effective_k(top_k, ranked.len());
    let top_index = index_of(ranked[0]);
    let mut draw = CanonDraw {
        path: PATH_ARGMAX,
        token: top_index,
        keep: 1,
        k_eff: u32::try_from(k_eff).expect("k_eff fits u32"),
        total_q: 0,
        thr_q: 0,
        mass_q: 0,
        target_q: 0,
        topk_rank_key: 0,
    };
    if k_eff == 1 || params.top_p_m == 0 {
        return draw;
    }
    let top = f64::from(logits[top_index as usize]);
    let weights: Vec<u64> = ranked[..k_eff]
        .iter()
        .map(|&key| {
            let logit = logits[index_of(key) as usize];
            weight_q(scaled_gap(logit, top, params.inv_t))
        })
        .collect();
    draw.path = PATH_DRAW;
    draw.total_q = weights.iter().sum();
    let shift = u32::try_from(params.top_p_e).expect("top_p_e is not negative");
    let product = u128::from(params.top_p_m) * u128::from(draw.total_q);
    draw.thr_q = narrow(ceil_shr(product, shift));
    let (keep, mass) = select(&weights, draw.thr_q).expect("thr_q never exceeds the total");
    draw.keep = u32::try_from(keep).expect("keep fits u32");
    draw.mass_q = mass;
    draw.target_q = narrow(floor_shr(u128::from(unit) * u128::from(mass), UNIT_BITS));
    let (rank, _) = select(&weights[..keep], draw.target_q + 1).expect("target is below the mass");
    draw.token = index_of(ranked[rank - 1]);
    if k_eff < ranked.len() {
        draw.topk_rank_key = ranked[k_eff - 1];
    }
    draw
}
