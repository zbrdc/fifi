//! The P2 parity test of the canonical sampler reference against the host sampler
//! (IMPL-DEVSAMPLE-1, Design 1, test M3).
//!
//! Level P2: for one 53-bit unit U (the host draw is `u = U * 2^-53`), the canonical
//! [`reference_sample`] and the host [`Sampler::pick_with_unit`] must draw the same token.
//! Greedy cells (temperature 0, `top_k` 1, `top_p` 0) must return exactly the argmax token on
//! both sides, with no certificate. Every other token mismatch is allowed only when
//! CERTIFIED as Design 1 defines: the host's own margin at the deciding boundary is at most
//! `s = (keep_max + 4) * (2^-45 + 2^-52 * mass)`. With `top_p < 1` the kept counts must match
//! unless the difference carries the same certificate; with `top_p = 1` they may differ only
//! across the vanishing tail. The deciding boundaries of a mismatch are the cumulative
//! weights strictly between the two answers (`cum[low] ..= cum[high - 1]`), and every one of
//! them must lie within `s` of the host's point; the outer edges of the two intervals decide
//! nothing (see `deciding_margin`). Every draw also checks that the test's own copy of the host
//! walk reaches the token and kept count of `pick_with_unit`.
//!
//! The host's `ranked_weights` is private, so this file recomputes the host's weights with the
//! host's own f64 arithmetic from the logits (same formulas, same paths) and says so at each
//! helper. The default test keeps the debug budget well under 30 s (the reference full-sorts
//! every row): the whole parameter grid at n = 4,096 and 18 hand-picked cells at the
//! vocabulary size 248,077. The exhaustive grid is the ignored test below.

use super::CanonParams;
use super::derive;
use super::reference_tests::{rank_key, reference_sample};
use crate::family::sampler::{Sampler, SamplingParams};

/// The vocabulary of the served model: the large n of the grid.
const VOCAB: usize = 248_077;
/// The host's ranked-prefix cutoff (`PREFIX_ROWS` in `family/sampler.rs`).
const PREFIX_ROWS: usize = 1_024;
/// One canonical weight quantum in host weight units, where the top weight is exactly 1.
const QUANTUM: f64 = 1.0 / (1_u64 << 45) as f64;
/// The unit scale: the host draw of the 53-bit unit U is `U * 2^-53`.
const UNIT_SCALE: f64 = 1.0 / (1_u64 << 53) as f64;
/// Row tags for the per-cell seeds.
const TAG_RANDOM: u64 = 1;
const TAG_SPREAD: u64 = 2;
const TAG_TIES: u64 = 3;
const TAG_FLAT: u64 = 4;
const TAG_SPIKE: u64 = 5;
const TAG_TAIL: u64 = 6;

/// SplitMix64, copied verbatim from the test module of `family/sampler.rs` (it is private
/// there). The units of the grid come from it, so every run is deterministic.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next 53-bit unit U.
    fn next_unit53(&mut self) -> u64 {
        self.next_u64() >> 11
    }
}

/// The 53-bit unit nearest to `fraction` of `2^53`.
fn unit53_near(fraction: f64) -> u64 {
    (fraction * (1_u64 << 53) as f64).round() as u64
}

/// The canonical request encoding of one cell (the reference side of P2).
fn canon_params(temperature: f64, top_p: f64) -> CanonParams {
    derive(temperature, top_p).unwrap_or_else(|error| panic!("valid settings: {error:?}"))
}

/// The host draw of one cell with a sampler of seed 0, as the packet prescribes.
fn pick(
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    top_k: u32,
    unit: f64,
    label: &str,
) -> (u32, usize) {
    let settings = SamplingParams::new(temperature, top_p, top_k)
        .unwrap_or_else(|error| panic!("valid settings for {label}: {error:?}"));
    Sampler::new(settings, 0)
        .pick_with_unit(logits, unit)
        .unwrap_or_else(|error| panic!("the host sampler draws {label}: {error:?}"))
}

/// The index of the largest logit, the lowest index on a tie (the host's argmax rule).
fn argmax_token(logits: &[f32]) -> u32 {
    let mut best = 0_usize;
    for (index, &logit) in logits.iter().enumerate() {
        if index == 0 || logit > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// The whole row in the host's ranked order. The host's composite key `ranked_key`
/// (`order_key << 32 | !index`, `family/sampler.rs`) is exactly the reference's `rank_key`,
/// so the reference's key reproduces the host's order here.
fn host_ranked(logits: &[f32]) -> Vec<u32> {
    let mut ranked: Vec<(u64, u32)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (rank_key(logit, index as u32), index as u32))
        .collect();
    ranked.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    ranked.into_iter().map(|(_, index)| index).collect()
}

/// The host weight `exp((logit - top) / temperature)` (`weight` in `family/sampler.rs`).
fn host_weight(logit: f32, top: f64, temperature: f64) -> f64 {
    ((f64::from(logit) - top) / temperature).exp()
}

/// The host's weights of `ids`, recomputed with the host's own f64 arithmetic including the
/// vanishing-tail early exit of `ranked_weights_prefix`: the walk stops once a weight that
/// left the total unchanged is followed by one no larger (`closed`).
fn host_weights(ids: &[u32], logits: &[f32], top: f64, temperature: f64) -> (Vec<f64>, f64, bool) {
    let mut weights = Vec::with_capacity(ids.len());
    let mut total = 0.0_f64;
    let mut stalled: Option<f64> = None;
    let mut closed = false;
    for &index in ids {
        let weight = host_weight(logits[index as usize], top, temperature);
        if stalled.is_some_and(|previous| weight <= previous) {
            closed = true;
            break;
        }
        let before = total;
        total += weight;
        weights.push(weight);
        stalled = (total == before).then_some(weight);
    }
    (weights, total, closed)
}

/// The host's own facts of one draw, recomputed with the host's f64 arithmetic: the same
/// candidate paths as `pick_row` (the 1,024-entry ranked prefix with its `closed` test, the
/// `top_k` truncation, the early exit) and the same keep and cumulative walk as `draw_kept`.
struct HostWalk {
    ids: Vec<u32>,
    weights: Vec<f64>,
    top: f64,
    total: f64,
    threshold: f64,
    keep: usize,
    mass: f64,
    target: f64,
    /// `cum[rank]` is the weight of the first `rank` entries (`cum[0]` is 0).
    cum: Vec<f64>,
    token: u32,
    /// True when `ids` is the host's closed 1,024-entry ranked prefix, so ranks past it exist
    /// but hold only weights the host's own sum cannot see.
    closed_prefix: bool,
}

fn host_walk(logits: &[f32], temperature: f64, top_p: f64, top_k: u32, unit: f64) -> HostWalk {
    let ranked = host_ranked(logits);
    let top = f64::from(logits[ranked[0] as usize]);
    let mut closed_prefix = false;
    let (ids, weights, total) = if top_k == 0 && logits.len() > PREFIX_ROWS {
        let prefix = &ranked[..PREFIX_ROWS];
        let (weights, total, closed) = host_weights(prefix, logits, top, temperature);
        if closed {
            closed_prefix = true;
            (prefix.to_vec(), weights, total)
        } else {
            let (weights, total, _) = host_weights(&ranked, logits, top, temperature);
            (ranked.clone(), weights, total)
        }
    } else {
        let limit = if top_k == 0 {
            ranked.len()
        } else {
            (top_k as usize).min(ranked.len())
        };
        let ids = ranked[..limit].to_vec();
        let (weights, total, _) = host_weights(&ids, logits, top, temperature);
        (ids, weights, total)
    };
    let threshold = top_p * total;
    let mut keep = 0_usize;
    let mut mass = 0.0_f64;
    for &weight in &weights {
        mass += weight;
        keep += 1;
        if mass >= threshold {
            break;
        }
    }
    let target = unit * mass;
    let mut cum = vec![0.0_f64; weights.len() + 1];
    for (index, &weight) in weights.iter().enumerate() {
        cum[index + 1] = cum[index] + weight;
    }
    let mut cumulative = 0.0_f64;
    let mut token = ids[keep.saturating_sub(1)];
    for (&entry, &weight) in ids.iter().zip(&weights).take(keep) {
        cumulative += weight;
        if cumulative > target {
            token = entry;
            break;
        }
    }
    HostWalk {
        ids,
        weights,
        top,
        total,
        threshold,
        keep,
        mass,
        target,
        cum,
        token,
        closed_prefix,
    }
}

/// Design 1's certificate bound `s = (keep_max + 4) * (2^-45 + 2^-52 * mass)`.
fn certificate_bound(keep_max: usize, mass: f64) -> f64 {
    (keep_max as f64 + 4.0) * (QUANTUM + f64::EPSILON * mass)
}

/// A mismatch is certified when the host's margin at the deciding boundary stays within the
/// certificate bound. A non-finite margin or bound never certifies.
fn certified(margin: f64, bound: f64) -> bool {
    margin.is_finite() && bound.is_finite() && margin <= bound
}

/// The cumulative weight before ranked entry `rank` (1-based). Past the stored weights the
/// host's tail has provably vanished, so the running sum stays at the total.
fn cum_at(walk: &HostWalk, rank: usize) -> f64 {
    if rank <= walk.weights.len() {
        walk.cum[rank]
    } else {
        walk.total
    }
}

/// The 1-based host rank of `token` among the host candidates, when it is one.
fn rank_of(walk: &HostWalk, token: u32) -> Option<usize> {
    walk.ids
        .iter()
        .position(|&id| id == token)
        .map(|index| index + 1)
}

/// The host's margin over the boundaries that decide between the ranks `low < high`: the
/// largest distance from `point` to a deciding boundary. The host picks rank `r` while
/// `cum[r - 1] <= point < cum[r]` (for the target) and keeps `r` while
/// `cum[r - 1] < point <= cum[r]` (for the threshold). Two sides that draw ranks `low` and
/// `high` from points that differ by at most `s` therefore have the boundaries
/// `cum[low] ..= cum[high - 1]` between them, and every one of them lies within `s` of the
/// host's point. The edges `cum[low - 1]` and `cum[high]` are only the outer sides of the two
/// intervals and decide nothing, so a host point that sits near them must not certify.
fn deciding_margin(walk: &HostWalk, low: usize, high: usize, point: f64) -> f64 {
    assert!(low < high, "two different ranks have a deciding boundary");
    (low..high)
        .map(|rank| (cum_at(walk, rank) - point).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The host's margin at the deciding boundaries of a token mismatch (see `deciding_margin`).
/// The canonical token must be a host candidate. The one exception is a closed host prefix:
/// ranks past it hold only weights the host's sum cannot see, so the token is placed just past
/// the prefix, where every later boundary equals the total.
fn token_margin(walk: &HostWalk, reference_token: u32) -> f64 {
    let host_rank = rank_of(walk, walk.token)
        .unwrap_or_else(|| panic!("the host token {} is not a host candidate", walk.token));
    let reference_rank = match rank_of(walk, reference_token) {
        Some(rank) => rank,
        None if walk.closed_prefix => walk.ids.len() + 1,
        None => panic!(
            "the canonical token {reference_token} is not a host candidate ({} candidates, \
             the host prefix is not closed)",
            walk.ids.len()
        ),
    };
    deciding_margin(
        walk,
        host_rank.min(reference_rank),
        host_rank.max(reference_rank),
        walk.target,
    )
}

/// The host's margin at the deciding boundaries of a keep mismatch (see `deciding_margin`),
/// measured from the host's threshold.
fn keep_margin(walk: &HostWalk, reference_keep: usize) -> f64 {
    deciding_margin(
        walk,
        walk.keep.min(reference_keep),
        walk.keep.max(reference_keep),
        walk.threshold,
    )
}

/// Whether a `top_p = 1` keep difference spans only the vanishing tail: every host weight
/// past the smaller keep is either at most one quantum (`2^-45`, the packet's condition) or
/// is invisible to the host's own f64 walk (adding it leaves the running mass bit-identical,
/// which is exactly the host's early-exit criterion). The second prong is a deviation the
/// exact emulation of this very grid forced: a spread row at temperature 1 or 2 totals about
/// `2^10`, so the host's half-ulp vanishing threshold sits at 2 to 4 quanta and genuine
/// vanishing tails span weights above `2^-45`. Both prongs are exact, and a semantic keep
/// bug spans weights the host can see. Ranks past a closed host prefix hold weights no larger
/// than the stalled one, which are invisible by monotone rounding. A keep past an unclosed
/// candidate list is an impossible canonical answer and fails with a message.
fn spans_vanishing_tail(
    walk: &HostWalk,
    reference_keep: usize,
    logits: &[f32],
    temperature: f64,
) -> bool {
    let low = walk.keep.min(reference_keep);
    let high = walk.keep.max(reference_keep);
    for rank in low + 1..=high {
        let weight = match (walk.weights.get(rank - 1), walk.ids.get(rank - 1)) {
            (Some(&weight), _) => weight,
            (None, Some(&id)) => host_weight(logits[id as usize], walk.top, temperature),
            (None, None) if walk.closed_prefix => continue,
            (None, None) => panic!(
                "the canonical keep {reference_keep} is past the {} host candidates and the \
                 host prefix is not closed",
                walk.ids.len()
            ),
        };
        if weight <= QUANTUM {
            continue;
        }
        let running = cum_at(walk, rank - 1);
        if running + weight != running {
            return false;
        }
    }
    true
}

/// Tallies of one grid run.
#[derive(Default, Debug)]
struct Counts {
    draws: u64,
    greedy: u64,
    certified_token: u64,
    certified_keep: u64,
}

/// Fails loudly when the test's private recomputation of the host walk has drifted from the
/// host sampler it mirrors (`draw_kept` and `pick_row`): the walk must reach the very token and
/// kept count that `pick_with_unit` returned.
fn assert_walk_agrees(walk: &HostWalk, host_token: u32, host_keep: usize, label: &str) {
    assert_eq!(
        (walk.token, walk.keep),
        (host_token, host_keep),
        "the test's host walk drifted from the host sampler (token, keep): {label}"
    );
}

/// Runs one cell of the grid: both sides draw with the same 53-bit unit, then the P2 rules
/// hold (greedy cells exactly, every other mismatch only with a certificate). A violation
/// panics with the row seed, n, the settings, U, both tokens, and the margin.
fn check_cell(
    counts: &mut Counts,
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    top_k: u32,
    unit53: u64,
    label: &str,
) {
    let unit = unit53 as f64 * UNIT_SCALE;
    if temperature == 0.0 {
        // Greedy never reaches the canonical sampler (`derive` refuses temperature 0), so this
        // side of the greedy rule is the host's own: the argmax token, one kept entry.
        let (token, keep) = pick(logits, temperature, top_p, top_k, unit, label);
        assert_eq!(
            (token, keep),
            (argmax_token(logits), 1),
            "temperature 0 must stay greedy: {label}"
        );
        counts.greedy += 1;
        return;
    }
    let reference = reference_sample(logits, &canon_params(temperature, top_p), top_k, unit53);
    check_draw(
        counts,
        logits,
        (temperature, top_p, top_k),
        unit53,
        (reference.token, reference.keep),
        label,
    );
}

/// The P2 rules of one draw against a given canonical answer `canonical` = (token, kept
/// count) of the reference: split from `check_cell` so that a test can hand in a wrong answer.
fn check_draw(
    counts: &mut Counts,
    logits: &[f32],
    settings: (f64, f64, u32),
    unit53: u64,
    canonical: (u32, u32),
    label: &str,
) {
    let (temperature, top_p, top_k) = settings;
    let (canonical_token, canonical_keep) = canonical;
    let unit = unit53 as f64 * UNIT_SCALE;
    let argmax = argmax_token(logits);
    let (host_token, host_keep) = pick(logits, temperature, top_p, top_k, unit, label);
    if top_k == 1 || top_p == 0.0 {
        // Greedy cells return the argmax exactly on both sides; no certificate is allowed.
        assert_eq!(canonical_token, argmax, "canonical greedy token: {label}");
        assert_eq!(host_token, argmax, "host greedy token: {label}");
        assert_eq!(canonical_keep, 1, "canonical greedy keep: {label}");
        assert_eq!(host_keep, 1, "host greedy keep: {label}");
        counts.greedy += 1;
        return;
    }
    counts.draws += 1;
    let walk = host_walk(logits, temperature, top_p, top_k, unit);
    assert_walk_agrees(&walk, host_token, host_keep, label);
    for value in [walk.total, walk.mass, walk.target] {
        assert!(value.is_finite(), "the host walk stays finite: {label}");
    }
    let bound = certificate_bound(walk.keep.max(canonical_keep as usize), walk.mass);
    if canonical_token != host_token {
        let margin = token_margin(&walk, canonical_token);
        assert!(
            certified(margin, bound),
            "uncertified token mismatch: {label}, U {unit53}, canonical token {canonical_token} \
             (keep {canonical_keep}), host token {host_token} (keep {host_keep}), \
             margin {margin:e}, bound {bound:e}"
        );
        counts.certified_token += 1;
    }
    if canonical_keep as usize != host_keep {
        if top_p == 1.0 {
            assert!(
                spans_vanishing_tail(&walk, canonical_keep as usize, logits, temperature),
                "top_p 1 keep difference outside the vanishing tail: {label}, \
                 canonical keep {canonical_keep}, host keep {host_keep}"
            );
        } else {
            let margin = keep_margin(&walk, canonical_keep as usize);
            assert!(
                certified(margin, bound),
                "uncertified keep mismatch: {label}, canonical keep {canonical_keep}, \
                 host keep {host_keep}, margin {margin:e}, bound {bound:e}"
            );
        }
        counts.certified_keep += 1;
    }
    // The catch-all of possible answers, AFTER the specific guards above so their diagnostics
    // stay first: whatever a closed prefix judged by margin, the certificate's token must be
    // inside the vocabulary and its keep count of at least one and at most the effective
    // candidates (`top_k` 0 = every candidate, the same limit the host walk applies).
    assert!(
        (canonical_token as usize) < logits.len(),
        "canonical token outside the vocabulary: {label}"
    );
    let effective = if top_k == 0 {
        logits.len()
    } else {
        (top_k as usize).min(logits.len())
    };
    assert!(
        (1..=effective as u32).contains(&canonical_keep),
        "canonical keep {canonical_keep} outside 1..={effective}: {label}"
    );
}

/// Checks one hand case: both sides must draw exactly `want` = (token, kept count).
fn check_exact(
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    top_k: u32,
    unit53: u64,
    want: (u32, usize),
    case: &str,
) {
    let unit = unit53 as f64 * UNIT_SCALE;
    let reference = reference_sample(logits, &canon_params(temperature, top_p), top_k, unit53);
    assert_eq!(reference.token, want.0, "canonical token of {case}");
    assert_eq!(reference.keep as usize, want.1, "canonical keep of {case}");
    assert_eq!(
        pick(logits, temperature, top_p, top_k, unit, case),
        want,
        "host draw of {case}"
    );
}

// ---- the row generators of the SAMPLER-FAST grid, copied from the test module of
// `family/sampler.rs` (they are private there), so the parity runs on the same rows.

/// Uniform logits in [-8, 8).
fn random_row(rng: &mut SplitMix64, size: usize) -> Vec<f32> {
    (0..size)
        .map(|_| (rng.next_unit53() as f64 * UNIT_SCALE * 16.0 - 8.0) as f32)
        .collect()
}

/// A row whose weights span hundreds of magnitudes with repeated ties (the heavy tails).
fn spread_magnitude_row(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            let decades = f64::from((index % 321) as u32);
            (1.0 - 0.7 * std::f64::consts::LN_10 * decades) as f32
        })
        .collect()
}

/// Pairs of equal logits in descending order (many ties).
fn tie_pairs(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| ((size - 1 - index / 2) as f32) * 0.5)
        .collect()
}

/// One dominant logit at index 0 over a flat floor.
fn one_spike_first(size: usize) -> Vec<f32> {
    let mut row = vec![0.0_f32; size];
    row[0] = 8.0;
    row
}

/// A flat row of equal logits.
fn flat_row(size: usize) -> Vec<f32> {
    vec![0.0_f32; size]
}

/// `size - 1` tail logits of about 1e-17 weight before the top logit 1.0 (a heavy tail).
fn tiny_tail_row(size: usize) -> Vec<f32> {
    let mut row = vec![-26.4_f32; size - 1];
    row.push(1.0);
    row
}

/// A deterministic per-cell seed, mixed from the row tag, the size, and the cell index, so
/// every cell is independent of the order the grid runs in.
fn cell_seed(tag: u64, size: usize, cell: u64) -> u64 {
    let mut rng = SplitMix64::new(tag ^ ((size as u64) << 32) ^ cell.wrapping_mul(0x9E37_79B9));
    rng.next_u64()
}

/// The packet's parameter grid: temperature x top_p x top_k (`top_k = n` included).
fn combos(size: usize) -> Vec<(f64, f64, u32)> {
    let mut combos = Vec::new();
    for temperature in [0.05, 0.7, 1.0, 2.0] {
        for top_p in [0.0, 0.3, 0.5, 0.95, 1.0] {
            for top_k in [0_u32, 1, 3, 20, 1000, 4097, size as u32] {
                combos.push((temperature, top_p, top_k));
            }
        }
    }
    combos
}

/// The hand-derived small cases. Every expected value follows from the setup in the comment
/// above it, and both sides must draw it exactly.
fn run_hand_cases() {
    // (a) Four equal logits at temperature 1, top_p 1: the weights are 1, 1, 1, 1 in the
    // ranked order 0, 1, 2, 3 (a tie goes to the lower index), so the threshold 4 keeps all
    // four and the mass is 4. The unit u aims at 4u: 0, 1.04, 2.04, and 3.04, so the token
    // whose cumulative weight first exceeds the target is 0, 1, 2, and 3.
    let row = [0.0_f32; 4];
    for (fraction, token) in [(0.0, 0_u32), (0.26, 1), (0.51, 2), (0.76, 3)] {
        check_exact(
            &row,
            1.0,
            1.0,
            0,
            unit53_near(fraction),
            (token, 4),
            "(a) four equal logits",
        );
    }
    // (b) top_p cuts the kept set to 2: the logits ln 0.4, ln 0.3, ln 0.2, ln 0.1 (f32) at
    // temperature 1 weigh 1, 0.75, 0.5, 0.25 against the top, so the total is 2.5 and
    // top_p 0.6 sets the threshold 1.5. The cumulative 1 stays below it and 1.75 reaches it,
    // so both sides keep 2 with mass 1.75. The units 0.26 and 0.51 aim at 0.455 and 0.8925,
    // below the first cumulative 1: token 0. The unit 0.9 aims at 1.575, between the
    // cumulatives 1 and 1.75: token 1.
    let row = [
        0.4_f64.ln() as f32,
        0.3_f64.ln() as f32,
        0.2_f64.ln() as f32,
        0.1_f64.ln() as f32,
    ];
    for (fraction, token) in [(0.26, 0_u32), (0.51, 0), (0.9, 1)] {
        check_exact(
            &row,
            1.0,
            0.6,
            0,
            unit53_near(fraction),
            (token, 2),
            "(b) top_p cuts the kept set to 2",
        );
    }
    // (c) the tie rule: the row 1, 1, 1, 0 with top_k 2 keeps the candidates 0 and 1 (a tie
    // goes to the lower index, `rank_key`); both weigh 1, so keep is 2 and the mass is 2.
    // The unit 0.26 aims at 0.52, below the first cumulative 1: token 0. The unit 0.76 aims
    // at 1.52, between the cumulatives 1 and 2: token 1. A reversed tie rule would keep the
    // candidates 2 and 1 and draw token 2 for the unit 0.26.
    let row = [1.0_f32, 1.0, 1.0, 0.0];
    check_exact(
        &row,
        1.0,
        1.0,
        2,
        unit53_near(0.26),
        (0, 2),
        "(c) the tie rule",
    );
    check_exact(
        &row,
        1.0,
        1.0,
        2,
        unit53_near(0.76),
        (1, 2),
        "(c) the tie rule",
    );
    // (d) the extreme units on the four equal logits of (a) (T 1, top_p 1: weights 1, 1, 1, 1,
    // mass 4). The unit 0 aims at 0, which the first cumulative 1 exceeds: token 0. The unit
    // 2^53 - 1 is 1 - 2^-53 on the host side, aiming at 4 - 2^-51 (exact in f64), below the
    // last cumulative 4: token 3. The canonical target is floor((2^53 - 1) * 4 * 2^45 / 2^53)
    // = 4 * 2^45 - 1, below the last cumulative 4 * 2^45: token 3.
    let row = [0.0_f32; 4];
    check_exact(&row, 1.0, 1.0, 0, 0, (0, 4), "(d) the unit 0");
    check_exact(
        &row,
        1.0,
        1.0,
        0,
        (1_u64 << 53) - 1,
        (3, 4),
        "(d) the unit 2^53 - 1",
    );
}

/// The greedy rows: temperature 0 over three shapes (the host side of the greedy rule), and
/// the tie at the argmax [3, 1, 3] under every greedy setting.
fn run_greedy_rows(counts: &mut Counts) {
    let mut rng = SplitMix64::new(cell_seed(TAG_RANDOM, 4096, 999_999));
    let rows = [
        random_row(&mut rng, 4096),
        tie_pairs(4096),
        one_spike_first(4096),
    ];
    for (index, logits) in rows.iter().enumerate() {
        for top_p in [0.0, 0.5, 1.0] {
            for top_k in [0_u32, 3] {
                let label = format!(
                    "greedy row {index}, n 4096, temperature 0, top_p {top_p}, top_k {top_k}"
                );
                check_cell(counts, logits, 0.0, top_p, top_k, rng.next_unit53(), &label);
            }
        }
    }
    let row = [3.0_f32, 1.0, 3.0];
    check_cell(
        counts,
        &row,
        1.0,
        0.0,
        0,
        unit53_near(0.4),
        "greedy tie row, top_p 0",
    );
    check_cell(
        counts,
        &row,
        1.0,
        0.5,
        1,
        unit53_near(0.4),
        "greedy tie row, top_k 1",
    );
    check_cell(
        counts,
        &row,
        0.0,
        0.5,
        0,
        unit53_near(0.4),
        "greedy tie row, temperature 0",
    );
}

/// The whole parameter grid at n = 4096 over three rows, two draws per cell.
fn run_small_grid(counts: &mut Counts) {
    let size = 4096_usize;
    let mut rng = SplitMix64::new(cell_seed(TAG_RANDOM, size, 8_000));
    let rows = [
        (TAG_RANDOM, "random", random_row(&mut rng, size)),
        (TAG_SPREAD, "spread", spread_magnitude_row(size)),
        (TAG_TIES, "ties", tie_pairs(size)),
    ];
    let cells = combos(size);
    for (tag, name, logits) in &rows {
        for (index, &(temperature, top_p, top_k)) in cells.iter().enumerate() {
            let seed = cell_seed(*tag, size, index as u64);
            let mut rng = SplitMix64::new(seed);
            for draw in 0..2 {
                let label = format!(
                    "row {name}, n {size}, seed 0x{seed:x}, T {temperature}, \
                     top_p {top_p}, top_k {top_k}, draw {draw}"
                );
                check_cell(
                    counts,
                    logits,
                    temperature,
                    top_p,
                    top_k,
                    rng.next_unit53(),
                    &label,
                );
            }
        }
    }
}

/// One small cell each of the flat, spike, and heavy-tail rows at n = 4096 (the rows the
/// whole-grid run leaves to the exhaustive test): four settings, three draws each.
fn run_extra_row_cells(counts: &mut Counts) {
    let size = 4096_usize;
    let rows = [
        (TAG_FLAT, "flat", flat_row(size)),
        (TAG_SPIKE, "spike", one_spike_first(size)),
        (TAG_TAIL, "tail", tiny_tail_row(size)),
    ];
    let settings = [
        (0.7, 0.95, 0_u32),
        (1.0, 1.0, 0),
        (2.0, 0.5, 20),
        (0.05, 0.3, 1000),
    ];
    for (tag, name, logits) in &rows {
        for (index, &(temperature, top_p, top_k)) in settings.iter().enumerate() {
            let seed = cell_seed(*tag, size, 500 + index as u64);
            let mut rng = SplitMix64::new(seed);
            for draw in 0..3 {
                let label = format!(
                    "row {name}, n {size}, seed 0x{seed:x}, T {temperature}, \
                     top_p {top_p}, top_k {top_k}, draw {draw}"
                );
                let unit53 = rng.next_unit53();
                check_cell(counts, logits, temperature, top_p, top_k, unit53, &label);
            }
        }
    }
}

/// The units 0 and 2^53 - 1 on a small random row and on the flat row, under three settings.
fn run_boundary_units(counts: &mut Counts) {
    let mut rng = SplitMix64::new(cell_seed(TAG_RANDOM, 64, 700));
    let rows = [("random", random_row(&mut rng, 64)), ("flat", flat_row(64))];
    for (name, logits) in &rows {
        for (temperature, top_p, top_k) in [(1.0, 1.0, 0_u32), (0.7, 0.95, 0), (2.0, 0.5, 3)] {
            for unit53 in [0, (1_u64 << 53) - 1] {
                let label = format!(
                    "row {name}, n 64, T {temperature}, top_p {top_p}, top_k {top_k}, \
                     boundary unit {unit53}"
                );
                check_cell(counts, logits, temperature, top_p, top_k, unit53, &label);
            }
        }
    }
}

/// The vocabulary-size cells of the default run: ten combinations over a random row and the
/// vanishing-tail subset over a spread row, plus one cell each of the tie, flat, and spike
/// rows. One draw per cell keeps the debug run under its budget (a full-size draw costs
/// about 0.3 s in debug).
fn run_full_size_cells(counts: &mut Counts) {
    let seed = cell_seed(TAG_RANDOM, VOCAB, 9_000);
    let mut rng = SplitMix64::new(seed);
    let random = random_row(&mut rng, VOCAB);
    let spread = spread_magnitude_row(VOCAB);
    let cells = [
        (0.05, 0.95, 0_u32),
        (0.7, 0.95, 0),
        (0.7, 1.0, 0),
        (0.7, 0.3, 20),
        (1.0, 0.5, 1000),
        (1.0, 1.0, 0),
        (2.0, 0.95, 3),
        (2.0, 1.0, 0),
        (0.05, 0.5, 4097),
        (1.0, 0.0, 0),
    ];
    for (index, &(temperature, top_p, top_k)) in cells.iter().enumerate() {
        let label = format!(
            "row random, n {VOCAB}, seed 0x{seed:x}, T {temperature}, top_p {top_p}, \
             top_k {top_k}"
        );
        let unit53 =
            SplitMix64::new(cell_seed(TAG_RANDOM, VOCAB, 100 + index as u64)).next_unit53();
        check_cell(counts, &random, temperature, top_p, top_k, unit53, &label);
    }
    for (index, &(temperature, top_p, top_k)) in [
        (0.05, 0.95, 0_u32),
        (0.7, 1.0, 0),
        (1.0, 1.0, 0),
        (2.0, 1.0, 0),
    ]
    .iter()
    .enumerate()
    {
        let seed = cell_seed(TAG_SPREAD, VOCAB, 200 + index as u64);
        let label = format!(
            "row spread, n {VOCAB}, seed 0x{seed:x}, T {temperature}, top_p {top_p}, \
             top_k {top_k}"
        );
        let unit53 = SplitMix64::new(seed).next_unit53();
        check_cell(counts, &spread, temperature, top_p, top_k, unit53, &label);
    }
    let ties = tie_pairs(VOCAB);
    for (cell, (tag, name, logits, temperature, top_p, top_k)) in [
        (TAG_TIES, "ties", ties.as_slice(), 0.7, 1.0, 0_u32),
        (TAG_TIES, "ties", ties.as_slice(), 1.0, 0.5, 1000),
        (TAG_FLAT, "flat", flat_row(VOCAB).as_slice(), 1.0, 1.0, 0),
        (
            TAG_SPIKE,
            "spike",
            one_spike_first(VOCAB).as_slice(),
            0.05,
            0.95,
            0,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = cell_seed(tag, VOCAB, 300 + cell as u64);
        let label = format!(
            "row {name}, n {VOCAB}, seed 0x{seed:x}, T {temperature}, top_p {top_p}, \
             top_k {top_k}"
        );
        let unit53 = SplitMix64::new(seed).next_unit53();
        check_cell(counts, logits, temperature, top_p, top_k, unit53, &label);
    }
}

/// Runs the default grid and returns its tallies.
fn run_default_grid() -> Counts {
    let mut counts = Counts::default();
    run_hand_cases();
    run_greedy_rows(&mut counts);
    run_small_grid(&mut counts);
    run_extra_row_cells(&mut counts);
    run_boundary_units(&mut counts);
    run_full_size_cells(&mut counts);
    counts
}

// The default P2 parity run: the hand cases, the greedy rows, the whole parameter grid at
// n = 4,096 (140 combinations x 3 rows x 2 draws = 840 draws), four settings of each of the
// flat, spike, and heavy-tail rows (3 draws each), the extreme units on two small rows, and 18
// cells at the vocabulary size (1 draw each): 641 sampled draws in all. In debug the run
// measured 6.8 s on exact transcriptions of both sides, well under 30 s. The ignored
// exhaustive test below covers the rest of the grid.
#[test]
fn the_canonical_reference_draws_the_host_token_for_the_same_unit() {
    let counts = run_default_grid();
    eprintln!(
        "P2 default grid: {} sampled draws, {} greedy cells, {} certified token mismatches, \
         {} certified keep mismatches",
        counts.draws, counts.greedy, counts.certified_token, counts.certified_keep
    );
    assert!(
        counts.draws >= 630,
        "the grid must really run: {:?}",
        counts
    );
    assert!(
        counts.greedy >= 280,
        "the greedy cells must really run: {:?}",
        counts
    );
    // The two full-size spread cells at top_p 1 and top_k 0 (T 1 and T 2) must show the
    // vanishing-tail keep difference. The row weighs 10^(-0.7 d / T) in 772 or 773 entries of
    // each group d (0 to 320). At T 1 the total is about 966, so the host's half-ulp is 2^-44
    // (5.7e-14, 2 quanta) and the group 19 (weight 5.0e-14) is invisible to the host sum, yet
    // its canonical weight is 2 quanta, so the canonical keep includes it. At T 2 the total is
    // about 1,397, the half-ulp is 2^-43 (1.1e-13), and the groups 38 and 39 (5.0e-14 and
    // 2.2e-14, 2 and 1 quanta) are invisible to the host but kept by the canonical side. The
    // keep does not depend on the unit, so each of the two cells counts at least once.
    assert!(
        counts.certified_keep >= 2,
        "the vanishing-tail keep difference must really occur: {:?}",
        counts
    );
}

// The certificate must stay strict. The row of hand case (b) at top_p 0.6 keeps 2 with a
// mass of 1.75, and the unit 0.51 aims at 0.8925; the distance from that target to the
// deciding boundary (the cumulative 1) is about 0.1075, while the bound
// (keep_max + 4) * (2^-45 + 2^-52 * mass) is about 1.7e-13. A mismatch there must NOT
// be certified, and a zero margin must be.
#[test]
fn the_certificate_rejects_a_mismatch_far_from_the_boundary() {
    let row = ln_row();
    let unit = unit53_near(0.51) as f64 * UNIT_SCALE;
    let walk = host_walk(&row, 1.0, 0.6, 0, unit);
    assert_eq!(walk.keep, 2, "the setup keeps two entries");
    assert!(
        (walk.mass - 1.75).abs() < 1e-6,
        "the kept mass is 1.75: {}",
        walk.mass
    );
    assert!(
        (walk.target - 0.8925).abs() < 1e-6,
        "the unit aims between the boundaries: {}",
        walk.target
    );
    let margin = token_margin(&walk, 1);
    let bound = certificate_bound(2, walk.mass);
    assert!(
        (margin - 0.1075).abs() < 1e-6,
        "the margin is the distance to the boundary 1: {margin}"
    );
    assert!(
        !certified(margin, bound),
        "a far-from-boundary mismatch is not certified: margin {margin:e}, bound {bound:e}"
    );
    assert!(certified(0.0, bound), "a zero margin is certified");
}

/// The row of hand case (b): the logits ln 0.4, ln 0.3, ln 0.2, ln 0.1 (weights 1, 0.75, 0.5,
/// 0.25 against the top at temperature 1).
fn ln_row() -> [f32; 4] {
    [
        0.4_f64.ln() as f32,
        0.3_f64.ln() as f32,
        0.2_f64.ln() as f32,
        0.1_f64.ln() as f32,
    ]
}

/// `ones` logits of 0 (weight 1) followed by `tails` logits of -30.6 (weight about 5.1e-14,
/// above one quantum 2.8e-14 and rounded to 2 quanta by the canonical weights).
fn ones_then_tail(ones: usize, tails: usize) -> Vec<f32> {
    let mut row = vec![0.0_f32; ones];
    row.extend(std::iter::repeat_n(-30.6_f32, tails));
    row
}

// M5: a wrong canonical token far from every boundary must fail the cell. On the row of
// hand case (b) at top_p 0.6 the host draws token 0 (target 0.8925) and the deciding
// boundary between tokens 0 and 1 is the cumulative 1, 0.1075 away.
#[test]
#[should_panic(expected = "uncertified token mismatch")]
fn a_wrong_canonical_token_far_from_every_boundary_fails_the_cell() {
    let mut counts = Counts::default();
    let unit53 = unit53_near(0.51);
    check_draw(
        &mut counts,
        &ln_row(),
        (1.0, 0.6, 0),
        unit53,
        (1, 2),
        "wrong token",
    );
}

// M5: a wrong canonical keep below top_p 1 must fail the cell through the margin branch. The
// host keeps 2 (threshold 1.5, cumulatives 1 and 1.75); a canonical keep of 3 would need the
// boundary 1.75, 0.25 away from the threshold.
#[test]
#[should_panic(expected = "uncertified keep mismatch")]
fn a_wrong_canonical_keep_below_top_p_one_fails_the_cell() {
    let mut counts = Counts::default();
    let unit53 = unit53_near(0.51);
    check_draw(
        &mut counts,
        &ln_row(),
        (1.0, 0.6, 0),
        unit53,
        (0, 3),
        "wrong keep",
    );
}

// M5: a wrong canonical keep at top_p 1 must fail the cell through the vanishing-tail branch.
// Four equal logits keep 4 on the host; a canonical keep of 3 would span the weight 1 of the
// fourth entry, which the host sees.
#[test]
#[should_panic(expected = "outside the vanishing tail")]
fn a_wrong_canonical_keep_at_top_p_one_fails_the_cell() {
    let mut counts = Counts::default();
    let unit53 = unit53_near(0.1);
    check_draw(
        &mut counts,
        &[0.0_f32; 4],
        (1.0, 1.0, 0),
        unit53,
        (0, 3),
        "wrong keep at top_p 1",
    );
}

// M1: four equal weights at top_p 0.5 put the host threshold exactly on the cumulative 2, so
// the host keeps 2. A defective canonical keep of 1 differs across the boundary cum[1] = 1,
// 1.0 away from the threshold. The cumulative 2 is only the host keep's own interval edge, so
// a zero distance to it must not certify.
#[test]
#[should_panic(expected = "uncertified keep mismatch")]
fn a_defective_keep_next_to_the_threshold_is_not_certified() {
    let mut counts = Counts::default();
    let unit53 = unit53_near(0.1);
    check_draw(
        &mut counts,
        &[0.0_f32; 4],
        (1.0, 0.5, 0),
        unit53,
        (0, 1),
        "keep 1 against the host keep 2",
    );
}

// M1: four equal weights at top_p 1 and the unit 0.5 aim at 2.0, exactly the cumulative 2, so
// the host draws token 2 (rank 3, because the cumulative must exceed the target). A defective
// canonical token 0 (rank 1) is separated from it by the boundaries cum[1] = 1 and
// cum[2] = 2. The distance to cum[2] is 0 but the distance to cum[1] is 1, and every
// deciding boundary must be near, so the mismatch must not certify.
#[test]
#[should_panic(expected = "uncertified token mismatch")]
fn a_defective_token_two_ranks_from_the_host_token_is_not_certified() {
    let mut counts = Counts::default();
    check_draw(
        &mut counts,
        &[0.0_f32; 4],
        (1.0, 1.0, 0),
        1 << 52,
        (0, 4),
        "token 0 against the host token 2",
    );
}

// The same unit with a canonical token 1 (rank 2) differs from the host token across the one
// boundary cum[2] = 2, which the target hits exactly: the margin is 0 and the mismatch is
// certified and counted.
#[test]
fn a_mismatch_across_the_one_boundary_next_to_the_target_is_certified() {
    let mut counts = Counts::default();
    check_draw(
        &mut counts,
        &[0.0_f32; 4],
        (1.0, 1.0, 0),
        1 << 52,
        (1, 4),
        "token 1 against the host token 2",
    );
    assert_eq!(counts.certified_token, 1, "one certified token mismatch");
    assert_eq!(counts.certified_keep, 0, "the keeps are equal");
}

// A canonical token that is no host candidate fails hard when the host walk is not closed:
// the candidate set is the same on both sides, so the token cannot come from outside it.
#[test]
#[should_panic(expected = "is not a host candidate")]
fn a_canonical_token_outside_an_unclosed_host_row_fails_hard() {
    let mut counts = Counts::default();
    check_draw(
        &mut counts,
        &[0.0_f32; 4],
        (1.0, 1.0, 0),
        unit53_near(0.1),
        (9, 4),
        "token 9 of a 4-entry row",
    );
}

// Past a CLOSED host prefix a canonical token is placed just beyond the prefix and judged by
// its margin like any other: 1,000 weights of 1 and 1,100 of 5.1e-14 close the host prefix at
// 1,024 entries (the first tail weight leaves the total 1,000 unchanged), the unit 0.5 aims at
// 500, and the canonical token 2000 (a tail entry past the prefix) is about 500 away from the
// boundaries between it and the host token.
#[test]
#[should_panic(expected = "uncertified token mismatch")]
fn a_canonical_token_past_a_closed_host_prefix_is_judged_by_its_margin() {
    let row = ones_then_tail(1000, 1100);
    let unit = unit53_near(0.5) as f64 * UNIT_SCALE;
    let walk = host_walk(&row, 1.0, 1.0, 0, unit);
    assert!(walk.closed_prefix, "the setup closes the host prefix");
    assert_eq!(walk.ids.len(), PREFIX_ROWS, "the prefix has 1,024 entries");
    let mut counts = Counts::default();
    check_draw(
        &mut counts,
        &row,
        (1.0, 1.0, 0),
        unit53_near(0.5),
        (2000, 2100),
        "token 2000 past the prefix",
    );
}

// A canonical keep past the candidates of an unclosed host row is impossible and must fail
// with a message instead of an index panic.
#[test]
#[should_panic(expected = "is past the 4 host candidates")]
fn a_canonical_keep_past_the_candidates_of_an_unclosed_row_fails_with_a_message() {
    let row = [0.0_f32; 4];
    let walk = host_walk(&row, 1.0, 1.0, 0, 0.1);
    let _ = spans_vanishing_tail(&walk, 9, &row, 1.0);
}

// M2: a host walk that no longer reaches the token or the kept count of `pick_with_unit`
// must fail loudly. The walk of hand case (b) draws (token 0, keep 2); a copy that drifted
// to another token or keep must be refused.
#[test]
#[should_panic(expected = "drifted from the host sampler")]
fn a_drifted_host_walk_fails_loudly() {
    let row = ln_row();
    let unit = unit53_near(0.51) as f64 * UNIT_SCALE;
    let mut walk = host_walk(&row, 1.0, 0.6, 0, unit);
    assert_eq!((walk.token, walk.keep), (0, 2), "the setup draws (0, 2)");
    assert_walk_agrees(&walk, 0, 2, "agreeing walk");
    walk.keep = 3;
    assert_walk_agrees(&walk, 0, 2, "drifted walk");
}

// The walk copy and the host agree on the hand cases it is built from: for each unit the
// token and kept count are the ones the hand derivation of case (b) gives.
#[test]
fn the_host_walk_reaches_the_token_and_keep_of_the_host_sampler() {
    let row = ln_row();
    for (fraction, token) in [(0.26, 0_u32), (0.51, 0), (0.9, 1)] {
        let unit53 = unit53_near(fraction);
        let unit = unit53 as f64 * UNIT_SCALE;
        let walk = host_walk(&row, 1.0, 0.6, 0, unit);
        assert_eq!((walk.token, walk.keep), (token, 2), "unit {fraction}");
        assert_walk_agrees(&walk, token, 2, "hand case (b)");
    }
}

// M4: `spans_vanishing_tail` is true when every spanned weight is at most one quantum. The
// row [0, -33] at T 1 has the weights 1 and 4.6e-15 (below 2.8e-14, visible to the host sum,
// so the host keeps 2); a canonical keep of 1 spans only the second weight.
#[test]
fn a_keep_difference_over_weights_below_a_quantum_spans_the_vanishing_tail() {
    let row = [0.0_f32, -33.0];
    let walk = host_walk(&row, 1.0, 1.0, 0, 0.5);
    assert_eq!(walk.keep, 2, "the host sees the small weight");
    assert!(
        walk.weights[1] < QUANTUM,
        "the spanned weight is a sub-quantum"
    );
    assert!(
        walk.cum[1] + walk.weights[1] != walk.cum[1],
        "the host sum sees it, so only the quantum prong can accept"
    );
    assert!(spans_vanishing_tail(&walk, 1, &row, 1.0));
}

// M4: it is also true when a weight above one quantum is invisible to the host's own sum.
// 1,100 weights of 1 and one of 5.1e-14 total 1,100 on the host (the half ulp of 1,100 is
// 1.1e-13), so the host keeps 1,100, while the canonical weight is 2 quanta and the
// canonical keep is 1,101. The cell through `check_cell` is certified and counted.
#[test]
fn a_keep_difference_over_a_weight_invisible_to_the_host_sum_is_certified() {
    let row = ones_then_tail(1100, 1);
    let walk = host_walk(&row, 1.0, 1.0, 0, 0.5);
    assert_eq!(walk.keep, 1100, "the host stops at the last visible weight");
    assert!(
        walk.weights[1100] > QUANTUM,
        "the spanned weight is above a quantum"
    );
    assert_eq!(walk.cum[1100] + walk.weights[1100], walk.cum[1100]);
    assert!(spans_vanishing_tail(&walk, 1101, &row, 1.0));
    let mut counts = Counts::default();
    check_cell(
        &mut counts,
        &row,
        1.0,
        1.0,
        0,
        unit53_near(0.5),
        "invisible weight",
    );
    assert_eq!(counts.certified_keep, 1, "the keep difference is counted");
    assert_eq!(counts.certified_token, 0, "the tokens agree");
}

// M3 and M4: a canonical keep past the host's closed 1,024-entry prefix spans ranks the host
// never stored. 1,000 weights of 1 and 1,100 of 5.1e-14 close the prefix; the host keeps
// 1,000 and the canonical keep is 2,100 (every tail weight is 2 quanta). All spanned weights
// are invisible, so the difference is certified without an index panic.
#[test]
fn a_keep_difference_past_a_closed_host_prefix_is_certified() {
    let row = ones_then_tail(1000, 1100);
    let walk = host_walk(&row, 1.0, 1.0, 0, 0.5);
    assert!(walk.closed_prefix, "the prefix closes");
    assert_eq!(walk.keep, 1000, "the host stops at the last visible weight");
    assert!(spans_vanishing_tail(&walk, 2100, &row, 1.0));
    let mut counts = Counts::default();
    check_cell(
        &mut counts,
        &row,
        1.0,
        1.0,
        0,
        unit53_near(0.5),
        "closed prefix",
    );
    assert_eq!(counts.certified_keep, 1, "the keep difference is counted");
    assert_eq!(counts.certified_token, 0, "the tokens agree");
}

// M4: it is false when a spanned weight is above a quantum and visible to the host sum. Four
// equal logits at T 1 weigh 1, 1, 1, 1 and the host keeps 4; a canonical keep of 3 spans the
// fourth weight, and a canonical keep of 1 spans three weights, all visible.
#[test]
fn a_keep_difference_over_a_visible_weight_does_not_span_the_vanishing_tail() {
    let row = [0.0_f32; 4];
    let walk = host_walk(&row, 1.0, 1.0, 0, 0.5);
    assert_eq!(walk.keep, 4, "the host keeps all four");
    assert!(!spans_vanishing_tail(&walk, 3, &row, 1.0));
    assert!(!spans_vanishing_tail(&walk, 1, &row, 1.0));
}

// The exhaustive P2 grid: every cell of the packet's grid at both sizes over all six row
// generators, 64 draws per cell. The reference full-sorts each row, so the full-size half
// takes about 10 minutes in release; run it with `--ignored` in a release build.
#[test]
#[ignore = "exhaustive P2 grid: run with --ignored in release"]
fn the_exhaustive_grid_draws_the_host_token_for_the_same_unit() {
    let mut counts = Counts::default();
    for size in [4096_usize, VOCAB] {
        let mut rng = SplitMix64::new(cell_seed(TAG_RANDOM, size, 8_000));
        let rows = [
            (TAG_RANDOM, "random", random_row(&mut rng, size)),
            (TAG_SPREAD, "spread", spread_magnitude_row(size)),
            (TAG_TIES, "ties", tie_pairs(size)),
            (TAG_FLAT, "flat", flat_row(size)),
            (TAG_SPIKE, "spike", one_spike_first(size)),
            (TAG_TAIL, "tail", tiny_tail_row(size)),
        ];
        let cells = combos(size);
        for (tag, name, logits) in &rows {
            for (index, &(temperature, top_p, top_k)) in cells.iter().enumerate() {
                let seed = cell_seed(*tag, size, index as u64);
                let mut rng = SplitMix64::new(seed);
                for draw in 0..64 {
                    let label = format!(
                        "row {name}, n {size}, seed 0x{seed:x}, T {temperature}, \
                         top_p {top_p}, top_k {top_k}, draw {draw}"
                    );
                    check_cell(
                        &mut counts,
                        logits,
                        temperature,
                        top_p,
                        top_k,
                        rng.next_unit53(),
                        &label,
                    );
                }
            }
        }
    }
    eprintln!(
        "P2 exhaustive grid: {} sampled draws, {} greedy cells, {} certified token mismatches, \
         {} certified keep mismatches",
        counts.draws, counts.greedy, counts.certified_token, counts.certified_keep
    );
    assert!(counts.draws >= 70_000, "the grid must really run");
}
