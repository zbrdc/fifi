//! Tight-buffer launches of the grouped GEMM: the device rows that `compute-sanitizer` runs to
//! prove the capacity rules (the `b1` rows of the q8_1 pad, the `b2` row of the fixup gather and
//! the per-expert tails). Each test builds the ids and the q8_1 columns the ordinary way in ample
//! buffers, then PROBES the launch wrapper for the smallest `y`, `ids_dst` and `expert_bounds`
//! ranges it accepts (a bisection on the plan function, the others held ample), copies the data
//! into device allocations of exactly those sizes, and launches over them. With the shipped rules
//! those sizes are the closed forms (the constant q8_1 pad, `ne_get_rows + ncols_max + 127` ids,
//! `n_experts + 1` bounds) and memcheck finds nothing; a mutant that lowers a rule lowers the
//! probed size, the launch then reads past its allocation, and memcheck reports it. Without a
//! sanitizer the tests check values only, so a capacity mutant dies in the sanitizer run alone:
//! that is the memcheck half of V1, V2, V3, V12, V23 and V24 (the refusal rows of the wrapper are
//! their CPU half).
//!
//! Every shape asserts what it is for before launching: the tile width of the J replica, the tails,
//! the empty experts, and for the `b2` row the host precondition (some stream-k boundary falls in
//! an empty top-expert tile whose ids gather reaches past `ne_get_rows + 126`, the earlier
//! capacity). The HARD GATE and the f64 reference follow.

use super::rig::{
    IdBufs, LaunchSpec, Mode, POISON, dequantized, errors, expert_blocks, f32_bytes, f32s,
    gate_dst, launch_mode, nan_bytes, normal_rows, reference,
};
use super::{device_facts, ints, read};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_mmid::{
    MmidIds, MmidIdsOut, MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
};
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange};

/// The frozen NMSE bound of a Q4_K expert gate/up launch, the class of every row here.
const GATE_UP_NMSE: f64 = 1e-3;

/// The smallest `size` in `1..=hi` for which `accepts(size)` holds. `accepts` must be monotone
/// and true at `hi`.
fn min_accepted(hi: usize, accepts: impl Fn(usize) -> bool) -> usize {
    let (mut refused, mut accepted) = (0, hi);
    while accepted - refused > 1 {
        let mid = refused + (accepted - refused) / 2;
        if accepts(mid) {
            accepted = mid;
        } else {
            refused = mid;
        }
    }
    accepted
}

/// An `MmidIds` built literally over the given ranges (only this module tree can), the way the
/// refusal rows of the launch build theirs.
fn literal_ids<'b>(
    shape: MmidShape,
    ids_src1: DeviceRange<'b>,
    ids_dst: DeviceRange<'b>,
    expert_bounds: DeviceRange<'b>,
) -> MmidIds<'b> {
    MmidIds {
        out: MmidIdsOut {
            ids_src1,
            ids_dst,
            expert_bounds,
        },
        shape,
    }
}

/// The launch call of `spec` over the given ranges.
fn launch_call<'b, 'i>(
    spec: &LaunchSpec<'_>,
    ids: &'i MmidIds<'b>,
    (weights, fixup): (&'b DeviceBuffer, &'b DeviceBuffer),
    (y, dst): (DeviceRange<'b>, DeviceRange<'b>),
) -> Result<MmidLaunchCall<'b, 'i>, EngError> {
    Ok(MmidLaunchCall {
        shape: spec.shape,
        weight: MmidWeight {
            base: weights.range(0, weights.bytes())?,
            quant: spec.quant,
            k: spec.k,
            n: spec.n,
            stride_bytes: spec.stride_bytes(),
        },
        ids,
        y,
        dst,
        ld: spec.ld,
        fixup: fixup.range(0, fixup.bytes())?,
    })
}

/// What a tight launch left on the device, and the sizes it ran at.
struct Tight {
    dst: Vec<f32>,
    /// The probed sizes in bytes: `y`, `ids_dst`, `expert_bounds`.
    sizes: (usize, usize, usize),
    bounds: Vec<i32>,
}

/// The ample build of `spec`, the probe, and the launch over exactly-sized `y`, `ids_dst` and
/// `expert_bounds` allocations (the weights and the fixup range are exact in every launch, and
/// `dst` is exactly `(ne - 1) * ld + n` floats).
fn tight_phase(weights: &[u8], spec: &LaunchSpec<'_>) -> Result<Tight, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (shape, ne) = (spec.shape, spec.shape.ne_get_rows());
    let bufs = IdBufs::new(&ctx, &shape, spec.rows, 0)?;
    let ids = bufs.build(&ctx, &stream, shape)?;
    let y_bytes = q8_1_bytes(spec.k, ne)?;
    let x_buf = ctx.upload(&f32_bytes(spec.x))?;
    let y_ample = ctx.upload(&vec![POISON; y_bytes])?;
    let quant = MmidQuantCall {
        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
        quant: spec.quant,
        k: spec.k,
        row_stride: spec.k,
        x: x_buf.range(0, spec.x.len() * 4)?,
        y: y_ample.range(0, y_bytes)?,
    };
    if spec.gather {
        ctx.mmq_quantize_gather_ids(&stream, &ids, quant)?;
    } else {
        ctx.mmq_quantize_scatter_ids(&stream, &ids, quant)?;
    }
    stream.synchronize()?;
    let y_host = read(&ctx, &y_ample)?;
    let ids_host = read(&ctx, &bufs.ids_dst)?;
    let bounds_host = read(&ctx, &bufs.bounds)?;

    let wbuf = ctx.upload(weights)?;
    let dst_floats = (ne - 1) * spec.ld + spec.n;
    let fixup = ctx.upload(&vec![0u8; fixup_floats_max(device_facts(&ctx)?.nsm)? * 4])?;
    let dst_ample = ctx.upload(&nan_bytes(dst_floats))?;
    let accepts = |y: usize, ids_dst: usize, bounds: usize| -> bool {
        let attempt = || -> Result<bool, EngError> {
            let literal = literal_ids(
                shape,
                bufs.ids_src1.range(0, ne * 4)?,
                bufs.ids_dst.range(0, ids_dst)?,
                bufs.bounds.range(0, bounds)?,
            );
            let call = launch_call(
                spec,
                &literal,
                (&wbuf, &fixup),
                (y_ample.range(0, y)?, dst_ample.range(0, dst_floats * 4)?),
            )?;
            Ok(ctx.plan_mmid_launch(&call).is_ok())
        };
        attempt().unwrap_or(false)
    };
    let (ids_full, bounds_full) = (bufs.ids_dst.bytes(), bufs.bounds.bytes());
    let sizes = (
        min_accepted(y_bytes, |c| accepts(c, ids_full, bounds_full)),
        min_accepted(ids_full, |c| accepts(y_bytes, c, bounds_full)),
        min_accepted(bounds_full, |c| accepts(y_bytes, ids_full, c)),
    );
    let y_t = ctx.upload(&y_host[..sizes.0])?;
    let ids_t = ctx.upload(&ids_host[..sizes.1])?;
    let bounds_t = ctx.upload(&bounds_host[..sizes.2])?;
    let dst_t = ctx.upload(&nan_bytes(dst_floats))?;
    let literal = literal_ids(
        shape,
        bufs.ids_src1.range(0, ne * 4)?,
        ids_t.range(0, sizes.1)?,
        bounds_t.range(0, sizes.2)?,
    );
    let call = launch_call(
        spec,
        &literal,
        (&wbuf, &fixup),
        (y_t.range(0, sizes.0)?, dst_t.range(0, dst_floats * 4)?),
    )?;
    ctx.mmq_mmid_launch(&stream, call)?;
    stream.synchronize()?;
    Ok(Tight {
        dst: f32s(&read(&ctx, &dst_t)?),
        sizes,
        bounds: ints(&bounds_host),
    })
}

/// The first stream-k block whose fixup gathers the ids of an empty TOP-expert tile past the old
/// capacity (mmq.cuh:1196-1330 replicated: block `b` starts at k-block `b * total / blocks`,
/// finishes the tile it starts in when it starts mid-tile and does not end inside it, and then
/// gathers `ids_dst[col_low + jt * J + j]` for all `j < J`, bounds or not): `(block, tile, last
/// index gathered)`, with the last index above `ne_get_rows + 126`.
fn b2_reach(
    mode: &Mode,
    kb: usize,
    shape: &MmidShape,
    bounds: &[i32],
) -> Option<(usize, usize, usize)> {
    let (e, ne) = (shape.n_experts, shape.ne_get_rows());
    let total = mode.ntiles * kb;
    let start = |b: usize| b * total / mode.blocks;
    (0..mode.blocks).find_map(|b| {
        let (kbc0, stop) = (start(b), start(b + 1));
        let inside = kbc0 / kb == stop / kb && stop % kb != 0;
        if kbc0 == stop || kbc0 % kb == 0 || inside {
            return None;
        }
        let tile = kbc0 / kb;
        let (jt, zt) = (tile % mode.ntx, tile / mode.ntx % e);
        let col_low = usize::try_from(bounds[zt]).ok()?;
        let last = col_low + jt * mode.j + mode.j - 1;
        (zt == e - 1 && bounds[zt] == bounds[zt + 1] && last > ne + 126).then_some((b, tile, last))
    })
}

/// Router rows over `shape` where each token draws `top_k` distinct experts from the first `pool`
/// with a seeded shuffle.
fn rows_from_pool(shape: &MmidShape, pool: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(shape.ne_get_rows());
    for _ in 0..shape.m {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < shape.top_k {
            let e = i32::try_from(rng.below(u64::try_from(pool).expect("small"))).expect("small");
            if !row.contains(&e) {
                row.push(e);
            }
        }
        rows.extend(row);
    }
    rows
}

/// [`rows_from_pool`] with the last slot of the last token replaced by `single`, an expert
/// outside the pool: it then holds exactly one column, the last of the compact order.
fn rows_with_single(shape: &MmidShape, pool: usize, single: i32, seed: u64) -> Vec<i32> {
    let mut rows = rows_from_pool(shape, pool, seed);
    let last = rows.len() - 1;
    rows[last] = single;
    rows
}

/// One tight row: the numbers, the launch, the HARD GATE, the reference, and a line for the
/// report. Returns the launch for the row's own assertions.
fn run_row(name: &str, spec: &LaunchSpec<'_>, weights: &[u8]) -> Tight {
    let tight = tight_phase(weights, spec).expect("the tight launch");
    let ne = spec.shape.ne_get_rows();
    let dims = (ne, spec.n, spec.ld);
    // The last row of dst holds n floats only (dst is tight): gate it with a padded copy.
    let mut padded = tight.dst.clone();
    padded.resize(
        ne * spec.ld + super::CANARY,
        f32::from_bits(super::rig::NAN_BITS),
    );
    gate_dst(&padded, dims, name);
    let deq = dequantized(spec.quant, weights);
    let err = errors(&padded, &reference(&deq, spec), dims);
    println!(
        "tight {name}: y {} B, ids_dst {} B, expert_bounds {} B; NMSE {:.3e}",
        tight.sizes.0, tight.sizes.1, tight.sizes.2, err.nmse
    );
    assert!(
        err.nmse <= GATE_UP_NMSE,
        "tight {name}: NMSE {:.3e} above the frozen {GATE_UP_NMSE:e}",
        err.nmse
    );
    tight
}

/// The facts the mode replica needs, read in its own context.
fn facts() -> (i32, usize, usize) {
    let phase = || -> Result<(i32, usize, usize), EngError> {
        let ctx = CudaContext::new()?;
        let f = device_facts(&ctx)?;
        Ok((f.cc, f.nsm, f.smpbo))
    };
    phase().expect("the device facts")
}

#[test]
fn mmq_mmid_tight_buffers_b1_single_token_top_expert_at_m_49() {
    if !testkit::device_present("mmq_mmid_tight_buffers_b1_single_token_top_expert_at_m_49") {
        return;
    }
    // The Qwen expert shape (Q4_K, N 512, K 2,048, fallback off), M 49. The unhinted chooser would
    // pick J = 64; with the tile hint on by default the launch aims at the average expert count and
    // picks J = 32, so the bounds asserted below hold for the narrower tile. The y-tile copies of
    // the last tile run past the last valid column when the highest expert holds one token: 880 B
    // more than the earlier, smaller pad.
    let shape = MmidShape {
        n_experts: 16,
        m: 49,
        top_k: 8,
    };
    let (k, n) = (2048, 512);
    let rows = rows_with_single(&shape, 15, 15, 0xb1);
    let weights = expert_blocks(QuantType::Q4K, 16, n, k, 0xb101);
    let x = normal_rows(shape.m, k, 0xb102);
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
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts());
    assert_eq!(mode.j, 64, "M 49 picks J 64: {mode:?}");
    let tight = run_row("B1 (M 49, single-token top expert)", &spec, &weights);
    assert_eq!(
        tight.bounds[16] - tight.bounds[15],
        1,
        "the highest expert holds one token"
    );
}

#[test]
fn mmq_mmid_tight_buffers_b1_wide_single_token_top_expert_at_m_225() {
    if !testkit::device_present("mmq_mmid_tight_buffers_b1_wide_single_token_top_expert_at_m_225") {
        return;
    }
    // The worst case of the constant pad: J = 128 (M 225..256, fallback off) and a one-column last
    // tile need S(128) - 144 = 18,288 B beyond the data, 144 B below the pad of 18,432.
    let shape = MmidShape {
        n_experts: 16,
        m: 225,
        top_k: 8,
    };
    let (k, n) = (256, 128);
    let rows = rows_with_single(&shape, 15, 15, 0xb2);
    let weights = expert_blocks(QuantType::Q4K, 16, n, k, 0xb201);
    let x = normal_rows(shape.m, k, 0xb202);
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
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts());
    assert_eq!(mode.j, 128, "M 225 picks J 128: {mode:?}");
    let tight = run_row(
        "B1 wide (M 225, J 128, single-token top expert)",
        &spec,
        &weights,
    );
    assert_eq!(
        tight.bounds[16] - tight.bounds[15],
        1,
        "the highest expert holds one token"
    );
}

#[test]
fn mmq_mmid_tight_buffers_b2_fixup_gathers_an_empty_top_expert() {
    if !testkit::device_present("mmq_mmid_tight_buffers_b2_fixup_gathers_an_empty_top_expert") {
        return;
    }
    // 8 experts, the highest empty, two column tiles (J 80 at M 130), K 2,048 (eight k-blocks), 64
    // tiles over 82 SMs: stream-k with the fixup, whose ids gather has no exit for an empty tile.
    let shape = MmidShape {
        n_experts: 8,
        m: 130,
        top_k: 2,
    };
    let (k, n) = (2048, 512);
    let rows = rows_from_pool(&shape, 7, 0xb3);
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0xb301);
    let x = normal_rows(shape.m, k, 0xb302);
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
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts());
    assert!(
        !mode.tiling && mode.fixup && mode.ntx >= 2,
        "the shape must run stream-k with the fixup over two column tiles: {mode:?}"
    );
    let tight = run_row("B2 (8 experts, empty top expert, fixup)", &spec, &weights);
    assert_eq!(tight.bounds[7], tight.bounds[8], "the top expert is empty");
    let reach = b2_reach(&mode, k / 256, &shape, &tight.bounds);
    let (block, tile, last) =
        reach.expect("a stream-k boundary in an empty top-expert tile past ne_get_rows + 126");
    println!(
        "tight B2: block {block} finishes tile {tile} and gathers ids_dst up to index {last} (ne_get_rows + 126 = {})",
        shape.ne_get_rows() + 126
    );
}

#[test]
fn mmq_mmid_tight_buffers_cover_expert_tails_with_a_hot_and_an_empty_expert() {
    if !testkit::device_present(
        "mmq_mmid_tight_buffers_cover_expert_tails_with_a_hot_and_an_empty_expert",
    ) {
        return;
    }
    // Per-expert tails `col_diff % J != 0`: expert 0 holds all 100 tokens (J 112, one tile with a
    // 12-column hole), experts 1..=5 hold 20 each, experts 6 and 7 are empty.
    let shape = MmidShape {
        n_experts: 8,
        m: 100,
        top_k: 2,
    };
    let (k, n) = (512, 256);
    let rows: Vec<i32> = (0..100).flat_map(|t| [0, 1 + t % 5]).collect();
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0xb4);
    let x = normal_rows(shape.m, k, 0xb402);
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
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts());
    assert_eq!(mode.j, 112, "M 100 picks J 112: {mode:?}");
    let tight = run_row(
        "tails (hot expert 0, experts 6 and 7 empty)",
        &spec,
        &weights,
    );
    let counts: Vec<i32> = tight.bounds[..9].windows(2).map(|w| w[1] - w[0]).collect();
    assert_eq!(
        counts,
        [100, 20, 20, 20, 20, 20, 0, 0],
        "the expert columns of the tails row"
    );
}

#[test]
fn mmq_mmid_tight_buffers_fallback_tail_at_m_24() {
    if !testkit::device_present("mmq_mmid_tight_buffers_fallback_tail_at_m_24") {
        return;
    }
    // The table-gap row: fallback on (N 160 is not a multiple of 128), M 24, so J is 32 where the
    // earlier upper bound J_ub is 24, and the last expert holds one column: the main kernel's ids
    // gather runs to `ne_get_rows + 30`, 8 ints past `ne_get_rows + J_ub - 1`.
    let shape = MmidShape {
        n_experts: 8,
        m: 24,
        top_k: 2,
    };
    let (k, n) = (256, 160);
    let rows: Vec<i32> = (0..24)
        .flat_map(|t| if t == 23 { [0, 7] } else { [0, 1 + t % 6] })
        .collect();
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0xb5);
    let x = normal_rows(shape.m, k, 0xb502);
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
    let mode = launch_mode(QuantType::Q4K, n, &shape, facts());
    assert_eq!(mode.j, 32, "M 24 under fallback picks J 32: {mode:?}");
    let tight = run_row(
        "fallback tail (N 160, M 24, single-token top expert)",
        &spec,
        &weights,
    );
    assert_eq!(
        tight.bounds[8] - tight.bounds[7],
        1,
        "the top expert holds one column"
    );
}
