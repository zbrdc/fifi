//! Pure tests of the canonical sampler reference and the request encoding.
//!
//! The known answers come from three independent sources: the published Random123 vectors,
//! standalone C programs (`philox_kat.c`, `exp_canon.c`, `canon_ref.c`, not part of this
//! repository), and hand arithmetic in the comments of each case.

use super::reference_tests::{
    CanonDraw, MAX_N, PATH_ARGMAX, PATH_DRAW, ceil_shr, exp_canon, floor_shr, philox4x32_10,
    rank_key, reference_sample, unit, value_key, weight_q,
};
use super::{CanonParams, derive};
use crate::EngError;

#[test]
fn the_temperature_bound_stays_one_contract() {
    // Two constants name the same contract bound (the sampler's and the encoding's); this
    // keeps them from drifting apart until C3a gives them one home.
    assert_eq!(
        crate::family::sampler::MAX_TEMPERATURE,
        super::MAX_TEMPERATURE
    );
}

const ONE_Q: u64 = 1 << 45;

fn params(temperature: f64, top_p: f64) -> CanonParams {
    derive(temperature, top_p).expect("valid request settings")
}

fn draw(row: &[f32], temperature: f64, top_p: f64, top_k: u32, unit: u64) -> CanonDraw {
    reference_sample(row, &params(temperature, top_p), top_k, unit)
}

/// The generator the C programs use to build rows: SplitMix64 from the golden-ratio constant.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new() -> Self {
        Self(0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A logit in [-8, 8) at the 2^53 scale.
    fn logit(&mut self) -> f32 {
        (((self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0) * 16.0 - 8.0) as f32
    }

    /// A multiple of 0.5 in [-1.5, 1.5], so a row of them holds many ties.
    fn tied_logit(&mut self) -> f32 {
        ((self.next_u64() % 7) as i32 - 3) as f32 * 0.5
    }
}

// ---- Philox ----

#[test]
fn philox_matches_the_random123_known_answers() {
    let cases: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0, 0, 0, 0],
            [0, 0],
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        ),
        (
            [u32::MAX; 4],
            [u32::MAX; 2],
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
        ),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ];
    for (counter, key, expected) in cases {
        assert_eq!(philox4x32_10(counter, key), expected, "{counter:x?}");
    }
}

#[test]
fn unit_reproduces_the_pinned_stream_vectors() {
    let vectors: [(u64, u64, u64); 15] = [
        (0, 0, 7_931_020_870_206_717),
        (0, 1, 3_261_426_827_205_785),
        (0, 31, 5_601_486_177_583_444),
        (42, 0, 4_220_652_138_765_662),
        (42, 1, 2_945_925_043_428_196),
        (42, 31, 800_005_672_551_909),
        (0x0123_4567_89ab_cdef, 0, 6_950_657_403_062_788),
        (0x0123_4567_89ab_cdef, 1, 2_893_646_986_066_242),
        (0x0123_4567_89ab_cdef, 31, 3_364_537_641_605_370),
        (0x5eed, 1, 5_938_320_525_554_641),
        (0x5eed, 2, 5_249_869_313_922_865),
        (0x5eed, 3, 1_954_318_033_083_160),
        // The position's high word is the counter's second word.
        (0x0123_4567_89ab_cdef, (1 << 32) + 5, 7_689_946_560_665_160),
        (42, (1 << 30) - 2, 1_840_595_053_805_041),
        (u64::MAX, u64::MAX, 2_358_833_809_302_298),
    ];
    for (seed, p, expected) in vectors {
        assert_eq!(unit(seed, p), expected, "seed {seed:#x} position {p}");
    }
}

#[test]
fn units_hold_53_bits_and_differ_across_positions_and_seeds() {
    let mut seen = Vec::new();
    for seed in [0_u64, 7, 1 << 40] {
        for p in 1000..1004 {
            let value = unit(seed, p);
            assert!(value < 1 << 53, "seed {seed:#x} position {p}");
            assert!(
                !seen.contains(&value),
                "seed {seed:#x} position {p} repeats"
            );
            seen.push(value);
        }
    }
}

// ---- exp_canon and weight_q ----

fn grid(i: u32) -> f64 {
    -32.0 + 32.0 * f64::from(i) / 100_000.0
}

#[test]
fn exp_canon_of_zero_is_exactly_one() {
    assert_eq!(exp_canon(0.0).to_bits(), 0x3FF0_0000_0000_0000);
}

#[test]
fn exp_canon_stays_within_two_ulp_of_libm_exp() {
    for i in 0..=100_000 {
        let x = grid(i);
        let distance = exp_canon(x).to_bits().abs_diff(x.exp().to_bits());
        assert!(distance <= 2, "x {x}: {distance} ulp");
    }
}

#[test]
fn exp_canon_is_non_decreasing_on_the_grid() {
    let mut previous = exp_canon(grid(0));
    for i in 1..=100_000 {
        let value = exp_canon(grid(i));
        assert!(value >= previous, "step {i} went down");
        previous = value;
    }
}

/// `exp_canon.c`'s function, hashed (FNV-1a over the result bits) on the same 100,001 points.
#[test]
fn exp_canon_matches_the_c_reference_bit_for_bit() {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for i in 0..=100_000 {
        for byte in exp_canon(grid(i)).to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    assert_eq!(hash, 4_410_109_205_753_002_096);
    let spots: [(f64, u64); 6] = [
        (-32.0, 0x3d0c_8464_f761_6468),
        (-31.89, 0x3d0f_d545_2a21_7b4e),
        (-10.0, 0x3f07_cd79_b564_7c9b),
        (-1.0, 0x3fd7_8b56_362c_ef38),
        (-0.5, 0x3fe3_68b2_fc6f_960a),
        (-1.0e-9, 0x3fef_ffff_ff76_8fa1),
    ];
    for (x, bits) in spots {
        assert_eq!(exp_canon(x).to_bits(), bits, "x {x}");
    }
}

#[test]
fn weight_q_pins_the_cutoff_the_top_and_the_c_values() {
    assert_eq!(weight_q(-32.0), 0);
    assert_eq!(weight_q(-31.89), 0);
    assert_eq!(weight_q(-31.88), 1);
    assert_eq!(weight_q(0.0), ONE_Q);
    assert_eq!(weight_q(-32.000_000_1), 0);
    assert_eq!(weight_q(f64::NEG_INFINITY), 0);
    assert_eq!(weight_q(-10.0), 1_597_368_022);
    assert_eq!(weight_q(-1.0), 12_943_607_142_008);
    assert_eq!(weight_q(-0.5), 21_340_400_414_614);
    assert_eq!(weight_q(-1.0e-9), 35_184_372_053_648);
}

// ---- 128-bit shifts ----

#[test]
fn shifts_match_u128_division_on_the_edges() {
    let mut random = SplitMix64::new();
    let mut values = vec![0, 1, (1_u128 << 116) - 1];
    for _ in 0..8 {
        let wide = (u128::from(random.next_u64()) << 64) | u128::from(random.next_u64());
        values.push(wide & ((1_u128 << 116) - 1));
    }
    for shift in [0_u32, 1, 52, 53, 63, 64, 65, 116, 127, 128, 1074] {
        for &value in &values {
            let (floor, ceil) = if shift >= 128 {
                (0, u128::from(value != 0))
            } else {
                let divisor = 1_u128 << shift;
                (
                    value / divisor,
                    value / divisor + u128::from(value % divisor != 0),
                )
            };
            assert_eq!(
                floor_shr(value, shift),
                floor,
                "floor {value:#x} >> {shift}"
            );
            assert_eq!(ceil_shr(value, shift), ceil, "ceil {value:#x} >> {shift}");
        }
    }
}

// ---- derive ----

/// `m * 2^-e`, built from exact power-of-two scalings so it does not share the encoder's bit
/// arithmetic. Each step is exact because the final value is a double and `m` has 53 bits.
fn rebuild(m: u64, e: i32) -> f64 {
    let mut value = m as f64;
    let mut left = e;
    while left > 0 {
        let step = left.min(500);
        let biased = u64::try_from(1023 - step).expect("a normal power of two");
        value *= f64::from_bits(biased << 52);
        left -= step;
    }
    value
}

#[test]
fn derive_reconstructs_top_p_exactly() {
    let values = [
        0.0,
        f64::from_bits(1),
        f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        f64::MIN_POSITIVE,
        2_f64.powi(-60),
        0.3,
        0.5,
        0.7,
        0.700_000_000_000_000_1,
        0.95,
        1.0 - f64::EPSILON / 2.0,
        1.0,
    ];
    for top_p in values {
        let canon = params(1.0, top_p);
        assert_eq!(
            rebuild(canon.top_p_m, canon.top_p_e).to_bits(),
            top_p.to_bits(),
            "top_p {top_p:e} gave ({}, {})",
            canon.top_p_m,
            canon.top_p_e
        );
    }
}

#[test]
fn derive_pins_the_decomposition_fields() {
    let cases: [(f64, u64, i32); 8] = [
        (1.0, 1 << 52, 52),
        (0.5, 1 << 52, 53),
        (0.75, 3 << 51, 53),
        (f64::MIN_POSITIVE, 1 << 52, 1074),
        (f64::from_bits(1), 1, 1074),
        (f64::from_bits(0x000F_FFFF_FFFF_FFFF), (1 << 52) - 1, 1074),
        (0.0, 0, 0),
        (-0.0, 0, 0),
    ];
    for (top_p, m, e) in cases {
        let canon = params(0.7, top_p);
        assert_eq!((canon.top_p_m, canon.top_p_e), (m, e), "top_p {top_p:e}");
    }
}

#[test]
fn derive_inv_t_is_the_f64_reciprocal() {
    for temperature in [0.05, 0.5, 0.7, 1.0, 1.7, 2.0] {
        let canon = params(temperature, 1.0);
        assert_eq!(canon.inv_t.to_bits(), (1.0 / temperature).to_bits());
    }
    assert_eq!(params(0.5, 1.0).inv_t, 2.0);
    assert_eq!(params(2.0, 1.0).inv_t, 0.5);
    assert_eq!(params(f64::from_bits(1), 1.0).inv_t, f64::INFINITY);
}

fn assert_bad_request(result: Result<CanonParams, EngError>, field: &str) {
    match result {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains(field), "message {message:?} names {field}");
        }
        other => panic!("expected a BadRequest naming {field}, got {other:?}"),
    }
}

#[test]
fn derive_refuses_temperatures_outside_zero_to_two() {
    for temperature in [
        0.0,
        -0.0,
        -0.1,
        2.0001,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_bad_request(derive(temperature, 0.9), "temperature");
    }
}

#[test]
fn derive_refuses_top_p_outside_zero_to_one() {
    let above_one = f64::from_bits(1.0_f64.to_bits() + 1);
    for top_p in [
        -0.1,
        1.0001,
        above_one,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_bad_request(derive(0.7, top_p), "top_p");
    }
}

// ---- value and rank keys ----

#[test]
fn value_key_orders_like_the_values_and_merges_the_zeros() {
    let ascending = [-3.4e38_f32, -1.0, -1.0e-40, -0.0, 0.0, 1.0e-40, 1.0, 3.4e38];
    let keys: Vec<u32> = ascending.iter().map(|&v| value_key(v)).collect();
    for (i, pair) in keys.windows(2).enumerate() {
        if i == 3 {
            assert_eq!(pair[0], pair[1], "-0.0 and 0.0 tie");
        } else {
            assert!(pair[0] < pair[1], "{pair:x?}");
        }
    }
    assert_eq!(value_key(0.0), 0x8000_0000);
    assert_eq!(value_key(1.0), 0xBF80_0000);
    assert_eq!(value_key(-1.0), 0x407F_FFFF);
    assert!(
        rank_key(2.0, 9) > rank_key(1.0, 0),
        "the larger value ranks first"
    );
    assert!(
        rank_key(1.0, 3) > rank_key(1.0, 4),
        "a tie goes to the lower index"
    );
}

// ---- reference_sample ----

const ZEROS4: [f32; 4] = [0.0; 4];

/// Case A: four equal logits, T 1, top_p 1, from the pinned Philox stream of seed 0x5eed.
#[test]
fn equal_logits_follow_the_pinned_draws() {
    let total = 4 * ONE_Q;
    let cases = [
        (1_u64, 2_u32, 92_786_258_211_791_u64),
        (2, 2, 82_029_208_030_044),
        (3, 0, 30_536_219_266_924),
    ];
    for (p, token, target) in cases {
        let got = draw(&ZEROS4, 1.0, 1.0, 0, unit(0x5eed, p));
        let want = CanonDraw {
            path: PATH_DRAW,
            token,
            keep: 4,
            k_eff: 4,
            total_q: total,
            thr_q: total,
            mass_q: total,
            target_q: target,
            topk_rank_key: 0,
        };
        assert_eq!(got, want, "position {p}");
    }
    assert_eq!(total, 140_737_488_355_328);
}

/// Case B: probabilities 0.4, 0.3, 0.2, 0.1 as f32 `ln` bits (what the host tests use).
#[test]
fn ln_probabilities_keep_the_host_nucleus_sizes() {
    let row = [0xbf6a_9207_u32, 0xbf9a_1bc8, 0xbfce_0210, 0xc013_5d8e].map(f32::from_bits);
    let u = unit(0x5eed, 1);
    assert_eq!(u, 5_938_320_525_554_641);
    // top_p 0.6 keeps {0.4, 0.3}: 0.7 reaches 0.6, and 0.4 alone does not.
    let got = draw(&row, 1.0, 0.6, 0, u);
    assert_eq!(
        (
            got.token,
            got.keep,
            got.total_q,
            got.thr_q,
            got.mass_q,
            got.target_q
        ),
        (
            1,
            2,
            87_960_927_432_204,
            52_776_556_459_323,
            61_572_650_005_458,
            40_593_987_209_481
        )
    );
    // top_p applies to the top_k mass: k 3 drops the 0.1 token from the total.
    let got = draw(&row, 1.0, 0.75, 3, u);
    assert_eq!(
        (
            got.token,
            got.keep,
            got.total_q,
            got.thr_q,
            got.mass_q,
            got.target_q
        ),
        (
            1,
            2,
            79_164_834_967_791,
            59_373_626_225_844,
            61_572_650_005_458,
            40_593_987_209_481
        )
    );
    assert_eq!(
        got.topk_rank_key,
        rank_key(row[2], 2),
        "the k-th ranked element"
    );
    let got = draw(&row, 1.0, 0.85, 0, u);
    assert_eq!(
        (
            got.token,
            got.keep,
            got.total_q,
            got.thr_q,
            got.mass_q,
            got.target_q
        ),
        (
            1,
            3,
            87_960_927_432_204,
            74_766_788_317_374,
            79_164_834_967_791,
            52_192_268_772_552
        )
    );
}

/// Case C: the nucleus boundary is inclusive. With four equal weights, top_p 0.5 needs exactly
/// two of them, so the cumulative mass equals the threshold at keep 2.
#[test]
fn the_top_p_boundary_is_inclusive() {
    let above = f64::from_bits(0.5_f64.to_bits() + 1);
    let below = f64::from_bits(0.5_f64.to_bits() - 1);
    let at = draw(&ZEROS4, 1.0, 0.5, 0, 0);
    assert_eq!((at.thr_q, at.keep, at.mass_q), (2 * ONE_Q, 2, 2 * ONE_Q));
    let over = draw(&ZEROS4, 1.0, above, 0, 0);
    assert_eq!(
        (over.thr_q, over.keep, over.mass_q),
        (2 * ONE_Q + 1, 3, 3 * ONE_Q)
    );
    let under = draw(&ZEROS4, 1.0, below, 0, 0);
    assert_eq!(
        (under.thr_q, under.keep, under.mass_q),
        (2 * ONE_Q, 2, 2 * ONE_Q)
    );
    // The smallest top_p still draws: one weight reaches a threshold of 1.
    let tiny = draw(&ZEROS4, 1.0, f64::from_bits(1), 0, 0);
    assert_eq!(
        (tiny.path, tiny.thr_q, tiny.keep, tiny.mass_q),
        (PATH_DRAW, 1, 1, ONE_Q)
    );
}

/// Case D: the token is the first one whose cumulative weight EXCEEDS the target.
#[test]
fn the_draw_boundary_is_exclusive() {
    let cases = [
        (0_u64, 0_u32, 0_u64),
        ((1 << 51) - 64, 0, ONE_Q - 1),
        (1 << 51, 1, ONE_Q),
        ((1 << 53) - 1, 3, 4 * ONE_Q - 1),
    ];
    for (u, token, target) in cases {
        let got = draw(&ZEROS4, 1.0, 1.0, 0, u);
        assert_eq!((got.token, got.target_q), (token, target), "unit {u}");
        assert_eq!(got.keep, 4);
    }
}

/// Case E: a whole vocabulary of equal logits with top_k 20 draws among indices 0..20.
#[test]
fn ties_go_to_the_lower_indices() {
    let row = vec![0.0_f32; 248_077];
    let got = draw(&row, 1.0, 1.0, 20, 5_938_320_525_554_641);
    let want = CanonDraw {
        path: PATH_DRAW,
        token: 13,
        keep: 20,
        k_eff: 20,
        total_q: 20 * ONE_Q,
        thr_q: 20 * ONE_Q,
        mass_q: 20 * ONE_Q,
        target_q: 463_931_291_058_956,
        topk_rank_key: rank_key(0.0, 19),
    };
    assert_eq!(got, want);
    assert_eq!(want.topk_rank_key, 0x8000_0000_FFFF_FFEC);
    for step in 0..64_u64 {
        let u = step * (1 << 47);
        let token = draw(&row, 1.0, 1.0, 20, u).token;
        assert_eq!(u64::from(token), (u * 20) >> 53, "unit {u}");
    }
}

#[test]
fn a_mix_of_signed_zeros_ranks_as_ties() {
    let row = [0.0_f32, -0.0, 0.0, -0.0];
    let half = 1 << 52;
    let upper = draw(&row, 1.0, 1.0, 2, half);
    assert_eq!(
        (upper.token, upper.target_q, upper.total_q),
        (1, ONE_Q, 2 * ONE_Q)
    );
    let lower = draw(&row, 1.0, 1.0, 2, half - 1);
    assert_eq!((lower.token, lower.target_q), (0, ONE_Q - 1));
    assert_eq!(upper.topk_rank_key, rank_key(-0.0, 1));
}

#[test]
fn top_k_cuts_a_tie_run_at_the_lower_indices() {
    let row = [5.0_f32, 1.0, 1.0, 1.0, 1.0];
    let low = draw(&row, 1.0, 1.0, 3, 1 << 52);
    let high = draw(&row, 1.0, 1.0, 3, (1 << 53) - 1);
    assert_eq!(
        (low.token, low.keep, low.total_q),
        (0, 3, 36_473_220_596_244)
    );
    assert_eq!((high.token, high.target_q), (2, 36_473_220_596_243));
    assert_eq!(
        high.topk_rank_key,
        rank_key(1.0, 2),
        "index 3 and 4 are cut"
    );
}

/// Case F: greedy-equivalent settings take the argmax and run no draw.
#[test]
fn top_p_zero_and_top_k_one_take_the_argmax() {
    let row = [1.0_f32, 3.0, 2.0, 3.0];
    let argmax_only = |k_eff: u32, got: CanonDraw| {
        let want = CanonDraw {
            path: PATH_ARGMAX,
            token: 1,
            keep: 1,
            k_eff,
            total_q: 0,
            thr_q: 0,
            mass_q: 0,
            target_q: 0,
            topk_rank_key: 0,
        };
        assert_eq!(got, want);
    };
    argmax_only(1, draw(&row, 0.7, 0.9, 1, 12_345));
    argmax_only(4, draw(&row, 0.7, 0.0, 0, 12_345));
    argmax_only(2, draw(&row, 0.7, 0.0, 2, (1 << 53) - 1));
    let single = draw(&[-4.0], 0.7, 0.9, 0, 77);
    assert_eq!(
        (single.path, single.token, single.keep, single.k_eff),
        (PATH_ARGMAX, 0, 1, 1)
    );
}

#[test]
fn top_k_at_or_above_the_row_length_means_the_whole_row() {
    let row = [0.5_f32, 0.25, -1.0, 2.0, 0.0];
    let base = draw(&row, 1.0, 1.0, 0, 1 << 51);
    assert_eq!(base.k_eff, 5);
    for top_k in [5, 6, u32::MAX] {
        assert_eq!(draw(&row, 1.0, 1.0, top_k, 1 << 51), base, "top_k {top_k}");
    }
    assert_eq!(draw(&row, 1.0, 1.0, 4, 1 << 51).k_eff, 4);
}

/// `d == 0` keeps `0 * inf` away: with T so small that 1/T is +inf, only the top weighs 2^45.
#[test]
fn an_underflowing_temperature_keeps_only_the_tied_top() {
    let row = [0.5_f32, 0.5, -1.0, 0.0];
    assert_eq!(params(f64::from_bits(1), 1.0).inv_t, f64::INFINITY);
    let got = draw(&row, f64::from_bits(1), 1.0, 0, 1 << 52);
    assert_eq!(
        (got.path, got.token, got.keep, got.total_q, got.target_q),
        (PATH_DRAW, 1, 2, 2 * ONE_Q, ONE_Q)
    );
}

#[derive(Clone, Copy)]
enum Row {
    Rand,
    Ties,
    Full,
}

/// (row, T, top_p, top_k, unit, [token, keep, total, thr, mass, target]) from `canon_ref.c`'s
/// functions on rows from `SplitMix64`.
type Golden = (Row, f64, f64, u32, u64, [u64; 6]);

#[rustfmt::skip]
const GOLDEN: [Golden; 39] = [
    (Row::Rand, 0.05, 0.3, 0, 7054479758228279, [155, 1, 39296797329248, 11789039198775, 35184372088832, 27556561555579]),
    (Row::Rand, 0.05, 0.3, 20, 6537175103922639, [155, 1, 39296797329219, 11789039198766, 35184372088832, 25535840249697]),
    (Row::Rand, 0.05, 0.3, 1000, 78040667738449, [155, 1, 39296797329248, 11789039198775, 35184372088832, 304846358353]),
    (Row::Rand, 0.05, 0.95, 0, 6733880905496958, [155, 2, 39296797329248, 37331957462786, 38792536362145, 29001725452820]),
    (Row::Rand, 0.05, 0.95, 20, 7590651673222228, [155, 2, 39296797329219, 37331957462759, 38792536362145, 32691697243275]),
    (Row::Rand, 0.05, 0.95, 1000, 7676615852728494, [155, 2, 39296797329248, 37331957462786, 38792536362145, 33061930926912]),
    (Row::Rand, 0.05, 1.0, 0, 5092128295659297, [155, 23, 39296797329248, 39296797329248, 39296797329248, 22216043850004]),
    (Row::Rand, 0.05, 1.0, 20, 2413598314505577, [155, 20, 39296797329219, 39296797329219, 39296797329219, 10530097216329]),
    (Row::Rand, 0.05, 1.0, 1000, 1785997171970385, [155, 23, 39296797329248, 39296797329248, 39296797329248, 7791985822961]),
    (Row::Rand, 0.7, 0.3, 0, 2428479697057449, [155, 4, 359615117790762, 107884535337229, 113323412626716, 30553738069066]),
    (Row::Rand, 0.7, 0.3, 20, 8309427614294494, [175, 4, 309188718296287, 92756615488887, 113323412626716, 104544450233059]),
    (Row::Rand, 0.7, 0.3, 1000, 7638393561046522, [175, 4, 359615117790762, 107884535337229, 113323412626716, 96101884819313]),
    (Row::Rand, 0.7, 0.95, 0, 8003692254130366, [130, 31, 359615117790762, 341634361901224, 342428041275215, 304277565538354]),
    (Row::Rand, 0.7, 0.95, 20, 4401672052692510, [112, 18, 309188718296287, 293729282381473, 297099435747033, 145187671129857]),
    (Row::Rand, 0.7, 0.95, 1000, 5565485599385222, [72, 31, 359615117790762, 341634361901224, 342428041275215, 211583898462086]),
    (Row::Rand, 0.7, 1.0, 0, 8438775401424326, [23, 200, 359615117790762, 359615117790762, 359615117790762, 336920625842229]),
    (Row::Rand, 0.7, 1.0, 20, 6176197526069134, [72, 20, 309188718296287, 309188718296287, 309188718296287, 212009365289091]),
    (Row::Rand, 0.7, 1.0, 1000, 3016522355364578, [143, 200, 359615117790762, 359615117790762, 359615117790762, 120435555100206]),
    (Row::Rand, 2.0, 0.3, 0, 3809671413216979, [143, 11, 988475587919520, 296542676375856, 318790389157090, 134835102236783]),
    (Row::Rand, 2.0, 0.3, 20, 5144505125190494, [2, 5, 513061739867643, 153918521960293, 160721775068600, 91797013942476]),
    (Row::Rand, 2.0, 0.3, 1000, 1286007972679732, [114, 11, 988475587919520, 296542676375856, 318790389157090, 45515478282986]),
    (Row::Rand, 2.0, 0.95, 0, 6078266355806140, [50, 73, 988475587919520, 939051808523544, 940464265275579, 634647035198387]),
    (Row::Rand, 2.0, 0.95, 20, 2886415999025354, [143, 19, 513061739867643, 487408652874261, 494597050310447, 158496864420467]),
    (Row::Rand, 2.0, 0.95, 1000, 3569449498071823, [69, 73, 988475587919520, 939051808523544, 940464265275579, 372695174682125]),
    (Row::Rand, 2.0, 1.0, 0, 5970444352833994, [97, 200, 988475587919520, 988475587919520, 988475587919520, 655213493661972]),
    (Row::Rand, 2.0, 1.0, 20, 6924782555525977, [28, 20, 513061739867643, 513061739867643, 513061739867643, 394444586564819]),
    (Row::Rand, 2.0, 1.0, 1000, 3300489006427673, [96, 200, 988475587919520, 988475587919520, 988475587919520, 362205022758133]),
    (Row::Ties, 0.05, 0.3, 20, 572371086539707, [13, 6, 703687441776640, 211106232532992, 211106232532992, 13414947340774]),
    (Row::Ties, 0.05, 0.95, 20, 6971183601772218, [101, 19, 703687441776640, 668503069687808, 668503069687808, 517392532944031]),
    (Row::Ties, 0.05, 1.0, 20, 5790437376338610, [96, 20, 703687441776640, 703687441776640, 703687441776640, 452377920026453]),
    (Row::Ties, 0.7, 0.3, 20, 6805489591752977, [40, 6, 703687441776640, 211106232532992, 211106232532992, 159503662306710]),
    (Row::Ties, 0.7, 0.95, 20, 2776094304160007, [54, 19, 703687441776640, 668503069687808, 668503069687808, 206038249136875]),
    (Row::Ties, 0.7, 1.0, 20, 7626871784134118, [112, 20, 703687441776640, 703687441776640, 703687441776640, 595849358135477]),
    (Row::Ties, 2.0, 0.3, 20, 6834550790479052, [40, 6, 703687441776640, 211106232532992, 211106232532992, 160184784151852]),
    (Row::Ties, 2.0, 0.95, 20, 3138755157012794, [67, 19, 703687441776640, 668503069687808, 668503069687808, 232954484309543]),
    (Row::Ties, 2.0, 1.0, 20, 8264577284715542, [126, 20, 703687441776640, 703687441776640, 703687441776640, 645670100368401]),
    (Row::Full, 0.7, 0.95, 0, 8193904301264572, [156449, 32402, 386344151468162556, 367026943894754412, 367027938660669907, 333887562628635575]),
    (Row::Full, 1.0, 1.0, 20, 3059810343997546, [165327, 20, 703300204365701, 703300204365701, 703300204365701, 238916135792273]),
    (Row::Full, 2.0, 1.0, 0, 6177685281777831, [133635, 248077, 1094287026499742994, 1094287026499742994, 1094287026499742994, 750528623433043231]),
];

#[test]
fn golden_draws_agree_with_the_c_reference() {
    let mut random = SplitMix64::new();
    let rand_row: Vec<f32> = (0..200).map(|_| random.logit()).collect();
    // The C generator draws one unit per (T, top_p, top_k) of the first 27 cases between rows.
    for _ in 0..27 {
        random.next_u64();
    }
    let ties_row: Vec<f32> = (0..200).map(|_| random.tied_logit()).collect();
    for _ in 0..9 {
        random.next_u64();
    }
    let full_row: Vec<f32> = (0..248_077).map(|_| random.logit()).collect();
    for (index, &(row, temperature, top_p, top_k, u, expected)) in GOLDEN.iter().enumerate() {
        let row = match row {
            Row::Rand => &rand_row,
            Row::Ties => &ties_row,
            Row::Full => &full_row,
        };
        let got = draw(row, temperature, top_p, top_k, u);
        let actual = [
            u64::from(got.token),
            u64::from(got.keep),
            got.total_q,
            got.thr_q,
            got.mass_q,
            got.target_q,
        ];
        assert_eq!(
            actual, expected,
            "golden case {index}: T {temperature} top_p {top_p} k {top_k}"
        );
    }
}

// ---- preconditions ----

#[test]
#[should_panic(expected = "logits are finite")]
fn the_reference_refuses_a_non_finite_row() {
    draw(&[0.0, f32::NAN, 1.0], 1.0, 1.0, 0, 0);
}

#[test]
#[should_panic(expected = "n is outside 1..=2^18")]
fn the_reference_refuses_an_empty_row() {
    draw(&[], 1.0, 1.0, 0, 0);
}

#[test]
#[should_panic(expected = "n is outside 1..=2^18")]
fn the_reference_refuses_a_row_above_the_cap() {
    draw(&vec![0.0; MAX_N + 1], 1.0, 1.0, 0, 0);
}

#[test]
#[should_panic(expected = "the unit holds 53 bits")]
fn the_reference_refuses_a_wide_unit() {
    draw(&ZEROS4, 1.0, 1.0, 0, 1 << 53);
}
