//! PG1b row invariance of the grouped launch (plan rev 3 section B.5 item 3, scope corrected by
//! MSG-85 M3). The claim is bit equality per FIXED tile configuration, where the configuration
//! includes the stream-k decomposition: a token's output rows must not depend on which compact
//! column it lands in. Two cases follow from the plan:
//!
//! - **Tiling mode** (the 256-expert resident launch, tile efficiency at least 90 %, no fixup):
//!   tokens permuted across expert boundaries and across J-column tiles keep their rows
//!   bit-equal. Asserted.
//! - **Stream-k mode** (a small slot bank, efficiency under 90 %, fixup running): a permutation
//!   inside one J-tile keeps the decomposition of every tile, so the rows stay bit-equal
//!   (asserted); a permutation across tiles changes which k split a column sees, so the sums may
//!   differ in the last bit: reported, never asserted.
//!
//! Each test first asserts that its shape really is in the mode it claims (the launcher's tiling
//! decision, mmq.cuh:1390-1434, replicated from the J replica of G1), so it cannot pass in the
//! wrong mode, and that two runs of the same inputs are bit-equal (determinism).

use super::device_facts;
use super::rig::{
    LaunchSpec, Launched, expert_blocks, gate_dst, launch_mode, launch_once, normal_rows,
};
use crate::EngError;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::prefill::mmq_mmid::MmidShape;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::{self, Rng};

/// `rows` and `x` with token `t` moved to position `sigma[t]`.
fn permute(
    rows: &[i32],
    x: &[f32],
    top_k: usize,
    k: usize,
    sigma: &[usize],
) -> (Vec<i32>, Vec<f32>) {
    let mut new_rows = vec![0; rows.len()];
    let mut new_x = vec![0.0; x.len()];
    for (t, &to) in sigma.iter().enumerate() {
        new_rows[to * top_k..(to + 1) * top_k].copy_from_slice(&rows[t * top_k..(t + 1) * top_k]);
        new_x[to * k..(to + 1) * k].copy_from_slice(&x[t * k..(t + 1) * k]);
    }
    (new_rows, new_x)
}

/// A random permutation of `0..n` (Fisher-Yates from one seed).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = usize::try_from(rng.below(u64::try_from(i + 1).expect("small"))).expect("small");
        p.swap(i, j);
    }
    p
}

/// How `got` differs from `base` after token `t`'s rows moved to `sigma[t]`: the number of
/// output values that are not bit-equal, and the largest absolute difference of those.
pub(super) fn row_differences(
    base: &Launched,
    got: &Launched,
    sigma: &[usize],
    (top_k, n, ld): (usize, usize, usize),
) -> (usize, f32) {
    let (mut differ, mut worst) = (0, 0.0f32);
    for (t, &to) in sigma.iter().enumerate() {
        for r in 0..top_k {
            let a = &base.dst[(t * top_k + r) * ld..][..n];
            let b = &got.dst[(to * top_k + r) * ld..][..n];
            for (x, y) in a.iter().zip(b) {
                if x.to_bits() != y.to_bits() {
                    differ += 1;
                    worst = worst.max((x - y).abs());
                }
            }
        }
    }
    (differ, worst)
}

/// The runs of one PG1b case and the device facts `(cc, nsm, smpbo)` they ran under.
pub(super) struct Runs {
    pub(super) base: Launched,
    pub(super) again: Launched,
    pub(super) moved: Vec<Launched>,
    pub(super) facts: (i32, usize, usize),
}

/// One base run, one repeat of it, and one run per permutation, over one weight buffer.
pub(super) fn runs(
    weights: &[u8],
    spec: &LaunchSpec<'_>,
    sigmas: &[Vec<usize>],
) -> Result<Runs, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let facts = device_facts(&ctx)?;
    let wbuf = ctx.upload(weights)?;
    let base = launch_once(&ctx, &stream, &wbuf, spec)?;
    let again = launch_once(&ctx, &stream, &wbuf, spec)?;
    let mut moved = Vec::new();
    for sigma in sigmas {
        let (rows, x) = permute(spec.rows, spec.x, spec.shape.top_k, spec.k, sigma);
        let permuted = LaunchSpec {
            rows: &rows,
            x: &x,
            ..*spec
        };
        moved.push(launch_once(&ctx, &stream, &wbuf, &permuted)?);
    }
    Ok(Runs {
        base,
        again,
        moved,
        facts: (facts.cc, facts.nsm, facts.smpbo),
    })
}

#[test]
fn mmq_mmid_pg1b_tiling_rows_survive_a_permutation_across_tiles_and_experts() {
    if !testkit::device_present(
        "mmq_mmid_pg1b_tiling_rows_survive_a_permutation_across_tiles_and_experts",
    ) {
        return;
    }
    // The resident shape: 256 experts, N 512, 130 tokens (J 80, two column tiles per expert).
    // Expert 7 sits in every token's ids, so its 130 columns cross the J boundary at 80; the
    // other seven ids are random, so tokens also move between experts' ranges.
    let shape = MmidShape {
        n_experts: 256,
        m: 130,
        top_k: 8,
    };
    let (k, n) = (512, 512);
    let mut rng = Rng::new(0x9b1b);
    let mut rows = Vec::with_capacity(shape.ne_get_rows());
    for _ in 0..shape.m {
        let mut row = vec![7];
        while row.len() < shape.top_k {
            let e = i32::try_from(rng.below(256)).expect("small");
            if !row.contains(&e) {
                row.push(e);
            }
        }
        rows.extend(row);
    }
    let weights = expert_blocks(QuantType::Q4K, 256, n, k, 0x9b1c);
    let x = normal_rows(shape.m, k, 0x9b1d);
    let spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld: n,
        rows: &rows,
        x: &x,
        gather: false,
    };
    let sigmas = vec![
        shuffled(shape.m, 1),
        (0..shape.m).rev().collect::<Vec<_>>(),
        (0..shape.m).map(|t| (t + 41) % shape.m).collect(),
    ];
    let Runs {
        base,
        again,
        moved,
        facts,
    } = runs(&weights, &spec, &sigmas).expect("the PG1b launches");
    let ne = shape.ne_get_rows();
    for (name, run) in std::iter::once(("base", &base))
        .chain(std::iter::once(("repeat", &again)))
        .chain(moved.iter().map(|r| ("permuted", r)))
    {
        gate_dst(&run.dst, (ne, n, n), name);
    }
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts);
    println!("PG1b tiling: {mode:?}");
    assert!(
        mode.tiling && !mode.fixup,
        "the shape is not in tiling mode: {mode:?}"
    );
    assert!(
        mode.ntx >= 2,
        "the shape has one column tile per expert: {mode:?}"
    );
    let hot = base.bounds[8] - base.bounds[7];
    assert_eq!(
        usize::try_from(hot).expect("gated"),
        shape.m,
        "expert 7 holds every token"
    );
    assert!(
        base.dst
            .iter()
            .zip(&again.dst)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "two runs differ"
    );
    for (sigma, run) in sigmas.iter().zip(&moved) {
        let (differ, worst) = row_differences(&base, run, sigma, (shape.top_k, n, n));
        println!(
            "PG1b tiling: permutation of token 0 to {}: {differ} values differ (max {worst:e})",
            sigma[0]
        );
        assert_eq!(
            differ, 0,
            "a permuted token's rows are not bit-equal in tiling mode"
        );
    }
}

#[test]
fn mmq_mmid_pg1b_stream_k_rows_survive_a_permutation_inside_a_tile() {
    if !testkit::device_present("mmq_mmid_pg1b_stream_k_rows_survive_a_permutation_inside_a_tile") {
        return;
    }
    // A slot bank: 8 experts, every token picks experts 0 and 1, so both hold 130 columns in two
    // J = 80 tiles, and the 64 tiles over 82 SMs run stream-k with the fixup (K 2,048 gives each
    // block a part of a tile's k range).
    let shape = MmidShape {
        n_experts: 8,
        m: 130,
        top_k: 2,
    };
    let (k, n) = (2048, 512);
    let rows: Vec<i32> = (0..shape.m)
        .flat_map(|t| if t % 2 == 0 { [0, 1] } else { [1, 0] })
        .collect();
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0x5ca1);
    let x = normal_rows(shape.m, k, 0x5ca2);
    let spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld: n,
        rows: &rows,
        x: &x,
        gather: false,
    };
    // Inside a tile: the columns 0..80 and 80..130 of an expert are its two tiles, and the
    // token order is the column order, so reversing each block permutes within a tile.
    let within: Vec<usize> = (0..80).rev().chain((80..130).rev()).collect();
    // Across tiles: a rotation by 41 moves tokens over the boundary at 80.
    let across: Vec<usize> = (0..shape.m).map(|t| (t + 41) % shape.m).collect();
    let Runs {
        base,
        again,
        moved,
        facts,
    } = runs(&weights, &spec, &[within.clone(), across.clone()]).expect("the PG1b launches");
    let ne = shape.ne_get_rows();
    for (name, run) in [
        ("base", &base),
        ("repeat", &again),
        ("within", &moved[0]),
        ("across", &moved[1]),
    ] {
        gate_dst(&run.dst, (ne, n, n), name);
    }
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts);
    println!("PG1b stream-k: {mode:?}");
    assert!(
        !mode.tiling && mode.fixup && mode.j == 80,
        "the shape is not stream-k with the fixup at J 80: {mode:?}"
    );
    assert!(
        base.dst
            .iter()
            .zip(&again.dst)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "two runs differ"
    );
    let dims = (shape.top_k, n, n);
    let (differ, worst) = row_differences(&base, &moved[0], &within, dims);
    println!("PG1b stream-k: inside a tile: {differ} values differ (max {worst:e})");
    assert_eq!(
        differ, 0,
        "a token moved inside its J-tile changed bits under stream-k"
    );
    let (differ, worst) = row_differences(&base, &moved[1], &across, dims);
    println!(
        "PG1b stream-k: across tiles (reported, not asserted): {differ} of {} values differ (max {worst:e})",
        ne * n
    );
}
