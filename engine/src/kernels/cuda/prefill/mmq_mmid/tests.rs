//! Seam-1 tests of the mmid wrappers (CPF-V4 G3; plan rev 3 sections B.5 items 5 and 8, C.2
//! N1). Device tests pin the ids guard N1 (the rewrite to `0..top_k`, the status masks of
//! MSG-85 M4 with literal words, the router range untouched, seam I1) and the builder (the
//! host sort of `mm_ids_helper` on random, skewed and poisoned ids, and the permutation
//! property). The refusal tables cover every wrapper guard, one row per rule, each asserted by
//! the text of its own error with every earlier check satisfied (the plan functions are
//! called directly, so no refused row reaches a launch).
//!
//! Device-test structure: every device phase is `Result`-only (no `assert`/`expect`/`unwrap`
//! while a device owner lives), drops every owner, then asserts on host `Vec`s. Every output
//! range is filled with `SENT` before the launch, and the HARD GATE asserts that every output
//! value lies in its domain before any other check, so an unwritten output fails. Canary ints
//! after every range must survive. The tests run with `--test-threads=1`.
//!
//! The grouped launch's device tests (the toy confirmation, Q8X, PG1b, the memcheck rows)
//! need G4's quantize entries and are the next part's. Here the launch is covered host-side:
//! the fill pin (finding 20's replacement), every number rule by tag, and every range rule
//! through the plan function. Its capacity rows construct an `MmidIds` literally, which only
//! this child module can do, because both constructors refuse those ranges first.
//!
//! Argued, not tested: the `shared memory` rule of the builder cannot fire on a real device
//! (4 * 2,048 bytes against the 48 KiB every built device has, FIFI_MIN_SMPBO); it exists so
//! that a raised chunk cap fails typed (mmid.cu:147).

use super::{
    GROUPED_TILE_MAX, HostIds, IdsGuardCall, MmidIds, MmidIdsOut, MmidLaunchCall, MmidLaunchShape,
    MmidShape, MmidWeight, check_host_ids, check_ids_shape, check_launch_shape, clamp_hint,
    fill_mmid_args, hint_cols, mmid_geometry, mmq_type_of, set_hint_override,
};
use crate::EngError;
use crate::kernels::cuda::native::prefill::mmq_mmid::MmidGeometry;
use crate::kernels::cuda::prefill::mmq_facts::MmqFacts;
use crate::kernels::cuda::prefill::mmq_size::MmqType;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

mod launch;
mod per_j;
mod pg1b;
mod poison;
mod quant;
mod rig;
mod small_m;
mod speed;
mod tight;

/// Fills every output before a launch: negative, so outside every id, slot and bound domain.
const SENT: i32 = 0xa5a5_a5a5_u32 as i32;
/// Canary ints after every output range.
const CANARY: usize = 64;
/// Mask 1 of the prefill status word: a router fault row (`route_experts`).
const ROUTER_FAULT: i32 = 1;
/// Mask 8 of the prefill status word: any other flagged row (`prefill_chunk`).
const BAD_ROW: i32 = 8;

fn qwen(m: usize) -> MmidShape {
    MmidShape {
        n_experts: 256,
        m,
        top_k: 8,
    }
}

fn le(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn ints(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect()
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>, EngError> {
    let mut out = vec![0u8; buffer.bytes()];
    ctx.download(buffer, &mut out)?;
    Ok(out)
}

/// The one place the tests read the device facts: `CudaContext::mmq_device_facts`. The raw-FFI
/// rows of the native child use it too.
pub(in crate::kernels::cuda) fn device_facts(ctx: &CudaContext) -> Result<MmqFacts, EngError> {
    ctx.mmq_device_facts()
}

fn as_i32(value: usize) -> i32 {
    i32::try_from(value).expect("the test indexes fit i32")
}

/// `m` rows of `top_k` distinct experts in `0..n_experts` (a partial Fisher-Yates per row).
fn random_rows(shape: &MmidShape, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let mut pool: Vec<i32> = (0..as_i32(shape.n_experts)).collect();
    let mut rows = Vec::with_capacity(shape.ne_get_rows());
    for _ in 0..shape.m {
        for r in 0..shape.top_k {
            let span = u64::try_from(shape.n_experts - r).expect("small");
            let j = r + usize::try_from(rng.below(span)).expect("small");
            pool.swap(r, j);
            rows.push(pool[r]);
        }
    }
    rows
}

/// The host guard: the guarded rows and the status bits a launch ORs.
fn host_guard(shape: &MmidShape, rows: &[i32]) -> (Vec<i32>, i32) {
    let e = as_i32(shape.n_experts);
    let mut out = Vec::with_capacity(rows.len());
    let mut bits = 0;
    for row in rows.chunks_exact(shape.top_k) {
        let flagged = row
            .iter()
            .enumerate()
            .any(|(s, &id)| !(0..e).contains(&id) || row[..s].contains(&id));
        if !flagged {
            out.extend_from_slice(row);
            continue;
        }
        bits |= if row.iter().all(|&id| id == -1) {
            ROUTER_FAULT
        } else {
            BAD_ROW
        };
        out.extend(0..as_i32(shape.top_k));
    }
    (out, bits)
}

/// The host sort of `mm_ids_helper` on guarded rows: `ids_dst` (by expert, then by token,
/// value `t * top_k + r`) and `expert_bounds`.
fn host_sort(shape: &MmidShape, guarded: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut ids_dst = Vec::with_capacity(guarded.len());
    let mut bounds = Vec::with_capacity(shape.n_experts + 1);
    for expert in 0..as_i32(shape.n_experts) {
        bounds.push(as_i32(ids_dst.len()));
        let slots = guarded.iter().enumerate().filter(|&(_, &id)| id == expert);
        ids_dst.extend(slots.map(|(slot, _)| as_i32(slot)));
    }
    bounds.push(as_i32(ids_dst.len()));
    (ids_dst, bounds)
}

/// The inverse of a permutation of `0..perm.len()`.
fn inverse(perm: &[i32]) -> Vec<i32> {
    let mut inv = vec![-1; perm.len()];
    for (c, &slot) in perm.iter().enumerate() {
        inv[usize::try_from(slot).expect("a slot")] = as_i32(c);
    }
    inv
}

/// What a guard launch left on the device.
struct GuardOut {
    ids_after: Vec<u8>,
    ids_g: Vec<i32>,
    status: Vec<i32>,
}

/// Runs the guard on `rows` with the status word preset to `status0` (`None`: no status
/// range in the call, the word stays `SENT`).
fn run_guard(shape: MmidShape, rows: &[i32], status0: Option<i32>) -> Result<GuardOut, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let ne = shape.ne_get_rows();
    let ids = ctx.upload(&le(rows))?;
    let ids_g = ctx.upload(&le(&vec![SENT; ne + CANARY]))?;
    let status = ctx.upload(&le(&[status0.unwrap_or(SENT), SENT, SENT, SENT]))?;
    let call = IdsGuardCall {
        shape,
        ids: ids.range(0, ne * 4)?,
        ids_g: ids_g.range(0, ne * 4)?,
        status: status0.map(|_| status.range(0, 4)).transpose()?,
    };
    ctx.mmq_ids_guard(&stream, call)?;
    stream.synchronize()?;
    Ok(GuardOut {
        ids_after: read(&ctx, &ids)?,
        ids_g: ints(&read(&ctx, &ids_g)?),
        status: ints(&read(&ctx, &status)?),
    })
}

/// The checks every guard launch passes: the HARD GATE on `ids_g`, the host guard's rows, the
/// canaries, the router range unchanged (I1) and the status word `status0 | bits`.
fn assert_guard(shape: &MmidShape, rows: &[i32], out: &GuardOut, status0: Option<i32>) {
    let ne = shape.ne_get_rows();
    let e = as_i32(shape.n_experts);
    assert!(
        out.ids_g[..ne].iter().all(|id| (0..e).contains(id)),
        "HARD GATE: ids_g holds a value outside 0..{e} (unwritten or out of range)"
    );
    let (want, bits) = host_guard(shape, rows);
    assert_eq!(out.ids_g[..ne], want[..], "guarded rows");
    assert!(out.ids_g[ne..].iter().all(|&v| v == SENT), "ids_g canary");
    assert_eq!(out.ids_after, le(rows), "the router ids changed (seam I1)");
    assert!(out.status[1..].iter().all(|&v| v == SENT), "status canary");
    assert_eq!(
        out.status[0],
        status0.map_or(SENT, |s| s | bits),
        "status word"
    );
}

/// The adversarial rows of plan rev 3 C.2 N1's tests, by name and row content, for `top_k` 8
/// over 256 experts.
fn poisoned_rows() -> Vec<(&'static str, [i32; 8])> {
    vec![
        ("all -1 (RD-5's fault form)", [-1; 8]),
        ("id 256", [1, 2, 3, 4, 256, 6, 7, 8]),
        ("id 300", [300, 2, 3, 4, 5, 6, 7, 8]),
        ("i32::MIN", [1, 2, 3, i32::MIN, 5, 6, 7, 8]),
        ("one -1 among valid ids", [9, 10, -1, 12, 13, 14, 15, 16]),
        ("one repeated id", [20, 21, 22, 23, 20, 25, 26, 27]),
        ("a fully repeated row", [5; 8]),
        ("i32::MAX", [i32::MAX, 1, 2, 3, 4, 5, 6, 7]),
    ]
}

/// Random clean Qwen rows with the poisoned rows written over rows 3, 7, 11, ...
fn mixed_rows(m: usize, seed: u64) -> Vec<i32> {
    let mut rows = random_rows(&qwen(m), seed);
    for (i, (_, row)) in poisoned_rows().iter().enumerate() {
        let at = (3 + 4 * i) * 8;
        rows[at..at + 8].copy_from_slice(row);
    }
    rows
}

#[test]
fn mmq_ids_guard_rewrites_flagged_rows_and_copies_clean_rows() {
    if !testkit::device_present("mmq_ids_guard_rewrites_flagged_rows_and_copies_clean_rows") {
        return;
    }
    let shape = qwen(41);
    let rows = mixed_rows(41, 0x6d6d_6964);
    let out = run_guard(shape, &rows, Some(0));
    let out = out.expect("the guard launch");
    assert_guard(&shape, &rows, &out, Some(0));
    // Literal pins, independent of the host guard: every poisoned row reads 0..8, a clean row
    // between them is unchanged, and the chunk with both kinds of row reads 9 (8 outranks 1,
    // book.rs:96-101, so it maps to prefill_chunk).
    for (i, (name, _)) in poisoned_rows().iter().enumerate() {
        let at = (3 + 4 * i) * 8;
        assert_eq!(out.ids_g[at..at + 8], [0, 1, 2, 3, 4, 5, 6, 7], "{name}");
        assert_eq!(
            out.ids_g[at + 8..at + 16],
            rows[at + 8..at + 16],
            "clean row after {name}"
        );
    }
    assert_eq!(out.status[0], ROUTER_FAULT | BAD_ROW);
}

#[test]
fn mmq_ids_guard_reports_mask_1_only_for_router_fault_rows() {
    if !testkit::device_present("mmq_ids_guard_reports_mask_1_only_for_router_fault_rows") {
        return;
    }
    let shape = qwen(17);
    let mut rows = random_rows(&shape, 0x0001);
    rows[0..8].copy_from_slice(&[-1; 8]);
    rows[16 * 8..17 * 8].copy_from_slice(&[-1; 8]);
    let out = run_guard(shape, &rows, Some(0)).expect("the guard launch");
    assert_guard(&shape, &rows, &out, Some(0));
    assert_eq!(
        out.status[0], ROUTER_FAULT,
        "an all -1 row is RD-5's fault: mask 1 only"
    );
    assert_eq!(out.ids_g[0..8], [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(out.ids_g[16 * 8..17 * 8], [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn mmq_ids_guard_reports_mask_8_for_every_other_flagged_row() {
    if !testkit::device_present("mmq_ids_guard_reports_mask_8_for_every_other_flagged_row") {
        return;
    }
    let shape = qwen(9);
    let cases: Vec<_> = poisoned_rows()
        .into_iter()
        .filter(|(_, row)| row != &[-1; 8])
        .collect();
    let base = random_rows(&shape, 0x0008);
    let runs: Result<Vec<_>, EngError> = cases
        .iter()
        .map(|(_, row)| {
            let mut rows = base.clone();
            rows[4 * 8..5 * 8].copy_from_slice(row);
            run_guard(shape, &rows, Some(0)).map(|out| (rows, out))
        })
        .collect();
    for ((name, _), (rows, out)) in cases.iter().zip(runs.expect("the guard launches")) {
        assert_guard(&shape, &rows, &out, Some(0));
        assert_eq!(out.status[0], BAD_ROW, "{name}: mask 8 only");
        assert_eq!(out.ids_g[4 * 8..5 * 8], [0, 1, 2, 3, 4, 5, 6, 7], "{name}");
    }
}

#[test]
fn mmq_ids_guard_leaves_the_status_word_alone_on_clean_rows() {
    if !testkit::device_present("mmq_ids_guard_leaves_the_status_word_alone_on_clean_rows") {
        return;
    }
    let shape = qwen(64);
    let rows = random_rows(&shape, 0x0040);
    // 0x40 names no mask: a launch that wrote the word instead of ORing it would lose it.
    let with_word = run_guard(shape, &rows, Some(0x40));
    let without = run_guard(shape, &mixed_rows(64, 0x0041), None);
    let (with_word, without) = (with_word.expect("clean"), without.expect("no status"));
    assert_guard(&shape, &rows, &with_word, Some(0x40));
    assert_eq!(with_word.status[0], 0x40);
    assert_eq!(with_word.ids_g[..shape.ne_get_rows()], rows[..]);
    assert_guard(&shape, &mixed_rows(64, 0x0041), &without, None);
}

#[test]
fn mmq_ids_guard_covers_every_top_k_width() {
    if !testkit::device_present("mmq_ids_guard_covers_every_top_k_width") {
        return;
    }
    let widths = [1usize, 2, 3, 6, 8, 10, 16, 32];
    let runs: Result<Vec<_>, EngError> = widths
        .iter()
        .map(|&top_k| {
            let shape = MmidShape {
                n_experts: 40,
                m: 9,
                top_k,
            };
            let mut rows = random_rows(&shape, 0x1000 + top_k as u64);
            rows[..top_k].fill(-1);
            rows[2 * top_k] = 40;
            if top_k > 1 {
                rows[5 * top_k + top_k - 1] = rows[5 * top_k];
            }
            run_guard(shape, &rows, Some(0)).map(|out| (shape, rows, out))
        })
        .collect();
    for (shape, rows, out) in runs.expect("the guard launches") {
        assert_guard(&shape, &rows, &out, Some(0));
        assert_eq!(
            out.status[0],
            ROUTER_FAULT | BAD_ROW,
            "top_k {}",
            shape.top_k
        );
    }
}

/// What a builder launch left on the device.
struct BuildOut {
    ids_after: Vec<u8>,
    ids_g: Vec<i32>,
    ids_src1: Vec<i32>,
    ids_dst: Vec<i32>,
    bounds: Vec<i32>,
    status: Vec<i32>,
    shape_back: MmidShape,
    /// The bytes of the ranges the returned `MmidIds` holds: ids_src1, ids_dst, expert_bounds.
    held: [usize; 3],
}

/// Runs the builder on `rows` with outputs sized exactly to the rules (the M2 form for
/// `ids_dst`), `SENT` in every output and canary, and the status word preset to 0.
fn run_build(shape: MmidShape, rows: &[i32]) -> Result<BuildOut, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
    let cap = ne + shape.m + 127;
    let ids = ctx.upload(&le(rows))?;
    let sent = |n: usize| ctx.upload(&le(&vec![SENT; n + CANARY]));
    let (ids_g, ids_src1, ids_dst, bounds) = (sent(ne)?, sent(ne)?, sent(cap)?, sent(e + 1)?);
    let status = ctx.upload(&le(&[0, SENT, SENT, SENT]))?;
    let guard = IdsGuardCall {
        shape,
        ids: ids.range(0, ne * 4)?,
        ids_g: ids_g.range(0, ne * 4)?,
        status: Some(status.range(0, 4)?),
    };
    let out = MmidIdsOut {
        ids_src1: ids_src1.range(0, ne * 4)?,
        ids_dst: ids_dst.range(0, cap * 4)?,
        expert_bounds: bounds.range(0, (e + 1) * 4)?,
    };
    let built = ctx.mmq_build_ids(&stream, guard, out)?;
    let shape_back = built.shape();
    let held = [
        built.out.ids_src1.bytes(),
        built.out.ids_dst.bytes(),
        built.out.expert_bounds.bytes(),
    ];
    stream.synchronize()?;
    Ok(BuildOut {
        ids_after: read(&ctx, &ids)?,
        ids_g: ints(&read(&ctx, &ids_g)?),
        ids_src1: ints(&read(&ctx, &ids_src1)?),
        ids_dst: ints(&read(&ctx, &ids_dst)?),
        bounds: ints(&read(&ctx, &bounds)?),
        status: ints(&read(&ctx, &status)?),
        shape_back,
        held,
    })
}

/// The checks every builder launch passes: the HARD GATE on every output, then the host guard
/// and the host sort, the permutation property stated directly, the per-expert count rule,
/// the untouched `ids_dst` pad and canaries, the router range and the status word.
fn assert_build(shape: &MmidShape, rows: &[i32], out: &BuildOut) {
    let (ne, e, m) = (shape.ne_get_rows(), shape.n_experts, shape.m);
    let n = as_i32(ne);
    assert!(
        out.ids_g[..ne].iter().all(|v| (0..as_i32(e)).contains(v)),
        "HARD GATE ids_g"
    );
    assert!(
        out.ids_dst[..ne].iter().all(|v| (0..n).contains(v)),
        "HARD GATE ids_dst"
    );
    assert!(
        out.ids_src1[..ne].iter().all(|v| (0..n).contains(v)),
        "HARD GATE ids_src1"
    );
    assert!(
        out.bounds[..=e].iter().all(|v| (0..=n).contains(v)),
        "HARD GATE bounds"
    );
    let (guarded, bits) = host_guard(shape, rows);
    let (want_dst, want_bounds) = host_sort(shape, &guarded);
    assert_eq!(out.ids_g[..ne], guarded[..], "guarded rows");
    assert_eq!(
        out.ids_dst[..ne],
        want_dst[..],
        "ids_dst against the host sort"
    );
    assert_eq!(
        out.bounds[..=e],
        want_bounds[..],
        "expert_bounds against the host sort"
    );
    assert_eq!(
        out.ids_src1[..ne],
        inverse(&want_dst)[..],
        "ids_src1 is the inverse map"
    );
    let mut seen = out.ids_src1[..ne].to_vec();
    seen.sort_unstable();
    assert!(
        seen.iter().copied().eq(0..n),
        "ids_src1 is a permutation of 0..{ne}"
    );
    for (c, &slot) in out.ids_dst[..ne].iter().enumerate() {
        let slot = usize::try_from(slot).expect("gated");
        assert_eq!(
            out.ids_src1[slot],
            as_i32(c),
            "ids_dst is the inverse of ids_src1"
        );
    }
    assert_eq!(out.bounds[0], 0);
    assert!(
        out.bounds[..=e]
            .windows(2)
            .all(|w| w[0] <= w[1] && w[1] - w[0] <= as_i32(m))
    );
    let cap = ne + m + 127;
    assert!(
        out.ids_dst[ne..].iter().all(|&v| v == SENT),
        "ids_dst pad and canary"
    );
    assert_eq!(out.ids_dst.len(), cap + CANARY);
    assert!(out.ids_g[ne..].iter().all(|&v| v == SENT), "ids_g canary");
    assert!(
        out.ids_src1[ne..].iter().all(|&v| v == SENT),
        "ids_src1 canary"
    );
    assert!(
        out.bounds[e + 1..].iter().all(|&v| v == SENT),
        "bounds canary"
    );
    assert_eq!(out.ids_after, le(rows), "the router ids changed (seam I1)");
    assert_eq!(
        out.status,
        [bits, SENT, SENT, SENT],
        "status word and canary"
    );
    assert_eq!(out.shape_back, *shape);
    assert_eq!(
        out.held,
        [ne * 4, cap * 4, (e + 1) * 4],
        "the MmidIds holds the given ranges"
    );
}

#[test]
fn mmq_build_ids_matches_the_host_sort_on_random_ids() {
    if !testkit::device_present("mmq_build_ids_matches_the_host_sort_on_random_ids") {
        return;
    }
    // Qwen at 1, 49 and 2,048 tokens; a small expert count; the generic helper path (top_k
    // 3, no compiled case); padded warp groups (6, 10); the widest case (32).
    let shapes = [
        qwen(1),
        qwen(49),
        qwen(2048),
        MmidShape {
            n_experts: 8,
            m: 64,
            top_k: 2,
        },
        MmidShape {
            n_experts: 16,
            m: 100,
            top_k: 3,
        },
        MmidShape {
            n_experts: 7,
            m: 33,
            top_k: 6,
        },
        MmidShape {
            n_experts: 300,
            m: 70,
            top_k: 10,
        },
        MmidShape {
            n_experts: 32,
            m: 5,
            top_k: 32,
        },
    ];
    let runs: Result<Vec<_>, EngError> = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let rows = random_rows(shape, 0x5eed + i as u64);
            run_build(*shape, &rows).map(|out| (rows, out))
        })
        .collect();
    for (shape, (rows, out)) in shapes.iter().zip(runs.expect("the builder launches")) {
        assert_build(shape, &rows, &out);
        assert_eq!(out.status[0], 0, "{shape:?}: clean rows set no mask");
    }
}

#[test]
fn mmq_build_ids_sorts_skewed_ids_with_empty_and_full_experts() {
    if !testkit::device_present("mmq_build_ids_sorts_skewed_ids_with_empty_and_full_experts") {
        return;
    }
    // Every token picks experts 0 and 1: two experts hold exactly m columns (the C2 maximum)
    // and the other six are empty; then one hot expert among 256 for the Qwen shape.
    let small = MmidShape {
        n_experts: 8,
        m: 64,
        top_k: 2,
    };
    let small_rows: Vec<i32> = (0..64).flat_map(|t| [t % 2, 1 - t % 2]).collect();
    let big = qwen(300);
    let mut big_rows = random_rows(&big, 0x407);
    for row in big_rows.as_chunks_mut::<8>().0.iter_mut() {
        if !row.contains(&255) {
            row[7] = 255;
        }
    }
    let runs =
        run_build(small, &small_rows).and_then(|a| run_build(big, &big_rows).map(|b| (a, b)));
    let (a, b) = runs.expect("the builder launches");
    assert_build(&small, &small_rows, &a);
    assert_eq!(a.bounds[..9], [0, 64, 128, 128, 128, 128, 128, 128, 128]);
    assert_build(&big, &big_rows, &b);
    assert_eq!(
        b.bounds[256] - b.bounds[255],
        300,
        "the hot expert holds every token"
    );
}

#[test]
fn mmq_build_ids_keeps_the_permutation_on_poisoned_ids() {
    if !testkit::device_present("mmq_build_ids_keeps_the_permutation_on_poisoned_ids") {
        return;
    }
    let shape = qwen(64);
    let rows = mixed_rows(64, 0x0bad);
    let out = run_build(shape, &rows).expect("the builder launch");
    assert_build(&shape, &rows, &out);
    assert_eq!(
        out.status[0],
        ROUTER_FAULT | BAD_ROW,
        "the combined chunk reads 9"
    );
    for (i, (name, _)) in poisoned_rows().iter().enumerate() {
        let at = (3 + 4 * i) * 8;
        assert_eq!(out.ids_g[at..at + 8], [0, 1, 2, 3, 4, 5, 6, 7], "{name}");
    }
}

#[test]
fn mmq_build_ids_reports_mask_1_for_router_fault_rows() {
    if !testkit::device_present("mmq_build_ids_reports_mask_1_for_router_fault_rows") {
        return;
    }
    let shape = qwen(33);
    let mut rows = random_rows(&shape, 0x00f1);
    for t in [0, 9, 32] {
        rows[t * 8..t * 8 + 8].fill(-1);
    }
    let out = run_build(shape, &rows).expect("the builder launch");
    assert_build(&shape, &rows, &out);
    assert_eq!(
        out.status[0], ROUTER_FAULT,
        "router faults alone read 1 (route_experts)"
    );
}

#[test]
fn mmq_type_of_pins_the_weight_type_mapping() {
    // The one mapping the typed quantize wrappers of part 2 call G4's entries through: every
    // glued weight type keeps its variant, so the q8_1 scale layout follows the weight
    // (mmq.cuh:61-101: Q4_K takes DS4, Q6_K and Q8_0 take D4). G4's own tests pin the ggml
    // discriminants of MmqType against the decode side.
    let rows = [
        (QuantType::Q80, MmqType::Q80),
        (QuantType::Q50, MmqType::Q50),
        (QuantType::Q4K, MmqType::Q4K),
        (QuantType::Q6K, MmqType::Q6K),
    ];
    for (quant, want) in rows {
        assert_eq!(mmq_type_of(quant), want, "{quant:?}");
    }
}

#[test]
fn mmq_mmid_refuses_id_shapes_out_of_range() {
    #[rustfmt::skip]
    let rows: [(&str, MmidShape); 7] = [
        ("n_experts", MmidShape { n_experts: 0, m: 1, top_k: 1 }),
        ("n_experts", MmidShape { n_experts: 65_536, m: 1, top_k: 1 }),
        ("m", MmidShape { n_experts: 256, m: 0, top_k: 8 }),
        ("m", MmidShape { n_experts: 256, m: 2049, top_k: 8 }),
        ("top_k", MmidShape { n_experts: 256, m: 1, top_k: 0 }),
        ("top_k", MmidShape { n_experts: 256, m: 1, top_k: 33 }),
        ("top_k", MmidShape { n_experts: 4, m: 1, top_k: 5 }),
    ];
    for (tag, shape) in rows {
        match check_ids_shape(&shape) {
            Err(EngError::BadRequest(text)) => {
                assert!(
                    text.starts_with(&format!("mmq_mmid: {tag}: ")),
                    "{shape:?}: {text}"
                );
            }
            other => panic!("{shape:?} was not refused by {tag}: {other:?}"),
        }
    }
    #[rustfmt::skip]
    let inside = [
        MmidShape { n_experts: 65_535, m: 2048, top_k: 32 },
        MmidShape { n_experts: 1, m: 1, top_k: 1 },
        MmidShape { n_experts: 4, m: 1, top_k: 4 },
    ];
    for shape in inside {
        assert!(
            check_ids_shape(&shape).is_ok(),
            "{shape:?} is inside the rules"
        );
    }
}

/// One range of a refusal row: buffer index, byte offset, bytes.
type Slot = (usize, usize, usize);
/// Slack after every range, so a shifted or short slot still builds a range.
const SLACK: usize = 4096;
/// The base slots of a shape, each range in its own buffer and sized exactly, in the order
/// ids, ids_g, status, ids_src1, ids_dst, expert_bounds.
fn base_slots(shape: &MmidShape) -> [Slot; 6] {
    let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
    [
        (0, 0, ne * 4),
        (1, 0, ne * 4),
        (2, 0, 4),
        (3, 0, ne * 4),
        (4, 0, (ne + shape.m + 127) * 4),
        (5, 0, (e + 1) * 4),
    ]
}

/// Plans the builder call the slots describe (no launch); `with_out` false plans the guard
/// call alone.
fn plan(
    ctx: &CudaContext,
    bufs: &[DeviceBuffer],
    shape: MmidShape,
    slots: &[Slot; 6],
    with_out: bool,
) -> Result<(), EngError> {
    let r = |i: usize| bufs[slots[i].0].range(slots[i].1, slots[i].2);
    let guard = IdsGuardCall {
        shape,
        ids: r(0)?,
        ids_g: r(1)?,
        status: Some(r(2)?),
    };
    if !with_out {
        return ctx.plan_ids_guard(&guard).map(|_| ());
    }
    let out = MmidIdsOut {
        ids_src1: r(3)?,
        ids_dst: r(4)?,
        expert_bounds: r(5)?,
    };
    ctx.plan_build_ids(&guard, &out).map(|_| ())
}

/// What planning one call gave.
type Planned = Result<(), EngError>;

/// One refusal row: a name, the slot edits, whether it plans the builder, and a needle of the
/// expected text.
type Row = (&'static str, Vec<(usize, Slot)>, bool, &'static str);

#[test]
fn mmq_ids_wrappers_refuse_ranges_that_do_not_fit() {
    if !testkit::device_present("mmq_ids_wrappers_refuse_ranges_that_do_not_fit") {
        return;
    }
    let shape = qwen(49);
    let base = base_slots(&shape);
    let (ne, e, cap) = (
        shape.ne_get_rows(),
        shape.n_experts,
        shape.ne_get_rows() + 49 + 127,
    );
    #[rustfmt::skip]
    let rows: Vec<Row> = vec![
        ("ids short", vec![(0, (0, 0, ne * 4 - 4))], false, "the CUDA ids range"),
        ("ids_g short", vec![(1, (1, 0, ne * 4 - 4))], false, "the CUDA ids_g range"),
        ("ids_g misaligned", vec![(1, (1, 2, ne * 4))], false, "ids_g range starts at offset 2"),
        ("status short", vec![(2, (2, 0, 2))], false, "the CUDA status range"),
        ("ids_g over ids (I1)", vec![(1, (0, 0, ne * 4))], false, "ids_g and ids ranges overlap"),
        ("status inside ids_g", vec![(2, (1, 8, 4))], false, "ids_g and status ranges overlap"),
        ("ids_src1 short", vec![(3, (3, 0, ne * 4 - 4))], true, "the CUDA ids_src1 range"),
        ("ids_dst one int short of M2", vec![(4, (4, 0, (cap - 1) * 4))], true, "the CUDA ids_dst range"),
        ("ids_dst at the rev 2 form", vec![(4, (4, 0, (ne + 127) * 4))], true, "the CUDA ids_dst range"),
        ("expert_bounds n_experts ints", vec![(5, (5, 0, e * 4))], true, "the CUDA expert_bounds range"),
        ("expert_bounds misaligned", vec![(5, (5, 2, (e + 1) * 4))], true, "expert_bounds range starts at offset 2"),
        ("ids_dst over ids_src1", vec![(4, (3, 0, cap * 4))], true, "ids_src1 and ids_dst ranges overlap"),
        ("expert_bounds over ids", vec![(5, (0, 0, (e + 1) * 4))], true, "expert_bounds and ids ranges overlap"),
        ("ids_src1 over ids_g", vec![(3, (1, 0, ne * 4))], true, "ids_g and ids_src1 ranges overlap"),
        ("status inside ids_dst", vec![(2, (4, 4 * ne, 4))], true, "status and ids_dst ranges overlap"),
    ];
    let phase = || -> Result<(Vec<Planned>, Planned), EngError> {
        let ctx = CudaContext::new()?;
        let bufs: Vec<DeviceBuffer> = base
            .iter()
            .map(|&(_, _, bytes)| ctx.upload(&vec![0u8; 2 * bytes + SLACK]))
            .collect::<Result<_, _>>()?;
        let results = rows
            .iter()
            .map(|(_, edits, with_out, _)| {
                let mut slots = base;
                for &(i, slot) in edits {
                    slots[i] = slot;
                }
                plan(&ctx, &bufs, shape, &slots, *with_out)
            })
            .collect();
        let base_ok = plan(&ctx, &bufs, shape, &base, true);
        Ok((results, base_ok))
    };
    let (results, base_ok) = phase().expect("the refusal phase");
    assert!(base_ok.is_ok(), "the base call plans: {base_ok:?}");
    for ((name, _, _, needle), result) in rows.iter().zip(results) {
        match result {
            Err(EngError::BadRequest(text)) => assert!(text.contains(needle), "{name}: {text}"),
            other => panic!("{name} was not refused: {other:?}"),
        }
    }
}

/// The small host shape of the `from_host` rows: 4 experts, 4 tokens, 2 ids, 8 columns.
fn small() -> MmidShape {
    MmidShape {
        n_experts: 4,
        m: 4,
        top_k: 2,
    }
}

#[test]
fn check_host_ids_derives_the_inverse_map() {
    // A rotation, not an involution, so the forward map differs from the inverse.
    let rotation = HostIds {
        ids_dst: &[1, 2, 3, 4, 5, 6, 7, 0],
        expert_bounds: &[0, 2, 4, 6, 8],
    };
    let inverse = check_host_ids(&small(), &rotation).expect("valid ids");
    assert_eq!(inverse, [7, 0, 1, 2, 3, 4, 5, 6]);
    // The builder's form: two experts full (m columns each), two empty.
    let sorted = HostIds {
        ids_dst: &[0, 2, 4, 6, 1, 3, 5, 7],
        expert_bounds: &[0, 4, 4, 8, 8],
    };
    let inverse = check_host_ids(&small(), &sorted).expect("valid ids");
    assert_eq!(inverse, [0, 4, 1, 5, 2, 6, 3, 7]);
}

#[test]
fn check_host_ids_refuses_every_c2_violation() {
    let good_dst = [0, 1, 2, 3, 4, 5, 6, 7];
    let good_bounds = [0, 2, 4, 6, 8];
    // Each row breaks one invariant and keeps every earlier one.
    #[rustfmt::skip]
    let rows: [(&str, &[i32], &[i32], &str); 10] = [
        ("ids_dst length", &[0, 1, 2, 3, 4, 5, 6], &good_bounds, "ids_dst holds 7 entries"),
        ("bounds length", &good_dst, &[0, 2, 4, 8], "expert_bounds 4, not 8 and 5"),
        ("bounds start above 0", &good_dst, &[1, 2, 4, 6, 8], "runs from 1 to 8, not from 0 to 8"),
        ("bounds end below", &good_dst, &[0, 2, 4, 6, 7], "runs from 0 to 7, not from 0 to 8"),
        ("bounds end above", &good_dst, &[0, 2, 4, 6, 9], "runs from 0 to 9, not from 0 to 8"),
        ("falling bounds", &good_dst, &[0, 4, 2, 6, 8], "falls from 4 to 2 at expert 1"),
        ("an expert with m + 1 columns", &good_dst, &[0, 5, 6, 7, 8], "expert 0 holds 5 columns"),
        ("an entry equal to m * top_k", &[0, 1, 2, 3, 4, 5, 6, 8], &good_bounds, "ids_dst[7] is 8, outside 0..8"),
        ("a negative entry", &[-1, 1, 2, 3, 4, 5, 6, 7], &good_bounds, "ids_dst[0] is -1, outside 0..8"),
        ("a duplicated entry", &[0, 1, 2, 3, 4, 5, 6, 6], &good_bounds, "ids_dst[7] repeats the token slot 6"),
    ];
    for (name, ids_dst, expert_bounds, needle) in rows {
        let host = HostIds {
            ids_dst,
            expert_bounds,
        };
        match check_host_ids(&small(), &host) {
            Err(EngError::BadRequest(text)) => {
                assert!(text.starts_with("mmq_mmid: ids values: "), "{name}: {text}");
                assert!(text.contains(needle), "{name}: {text}");
            }
            other => panic!("{name} was not refused: {other:?}"),
        }
    }
}

/// One `from_host` refusal row: a name, `ids_dst`, `expert_bounds`, whether the `ids_dst`
/// range is one int short, and a needle of the expected text.
type HostRow<'a> = (&'a str, &'a [i32], &'a [i32], bool, &'a str);

/// The three ranges `from_host` wrote, read back whole (ids_src1, ids_dst, expert_bounds, each
/// with its canary or pad), and the outcome of the call.
type HostOut = (Result<MmidShape, EngError>, [Vec<i32>; 3]);

/// Runs `from_host` into fresh `SENT` buffers sized exactly to the rules; `short_dst` makes
/// the `ids_dst` range one int short of rule M2.
fn run_from_host(
    ctx: &CudaContext,
    stream: &DeviceStream,
    shape: MmidShape,
    host: HostIds<'_>,
    short_dst: bool,
) -> Result<HostOut, EngError> {
    let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
    let cap = ne + shape.m + 127;
    let sent = |n: usize| ctx.upload(&le(&vec![SENT; n + CANARY]));
    let (src1, dst, bounds) = (sent(ne)?, sent(cap)?, sent(e + 1)?);
    let out = MmidIdsOut {
        ids_src1: src1.range(0, ne * 4)?,
        ids_dst: dst.range(0, (cap - usize::from(short_dst)) * 4)?,
        expert_bounds: bounds.range(0, (e + 1) * 4)?,
    };
    let made = MmidIds::from_host(stream, shape, host, out).map(|ids| ids.shape());
    let back = [
        ints(&read(ctx, &src1)?),
        ints(&read(ctx, &dst)?),
        ints(&read(ctx, &bounds)?),
    ];
    Ok((made, back))
}

#[test]
fn mmid_ids_from_host_uploads_what_the_builder_wrote() {
    if !testkit::device_present("mmid_ids_from_host_uploads_what_the_builder_wrote") {
        return;
    }
    // The Qwen shape and an 8-slot bank shape (seam I2's tiered caller).
    let shapes = [
        qwen(49),
        MmidShape {
            n_experts: 8,
            m: 64,
            top_k: 8,
        },
    ];
    // The builder runs in its own context, dropped before the from_host context opens.
    let phase = || -> Result<Vec<(BuildOut, HostOut)>, EngError> {
        let mut runs = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let rows = random_rows(shape, 0x4057 + i as u64);
            let built = run_build(*shape, &rows)?;
            let ne = shape.ne_get_rows();
            let host = HostIds {
                ids_dst: &built.ids_dst[..ne],
                expert_bounds: &built.bounds[..=shape.n_experts],
            };
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let uploaded = run_from_host(&ctx, &stream, *shape, host, false)?;
            runs.push((built, uploaded));
        }
        Ok(runs)
    };
    let runs = phase().expect("the builder and from_host phase");
    for (shape, (built, (made, [src1, dst, bounds]))) in shapes.iter().zip(runs) {
        let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
        let n = as_i32(ne);
        assert!(
            src1[..ne].iter().all(|v| (0..n).contains(v)),
            "HARD GATE ids_src1"
        );
        assert!(
            dst[..ne].iter().all(|v| (0..n).contains(v)),
            "HARD GATE ids_dst"
        );
        assert!(
            bounds[..=e].iter().all(|v| (0..=n).contains(v)),
            "HARD GATE bounds"
        );
        assert_eq!(made.expect("from_host accepts the builder's ids"), *shape);
        assert_eq!(dst[..ne], built.ids_dst[..ne], "ids_dst uploaded as given");
        assert_eq!(bounds[..=e], built.bounds[..=e], "bounds uploaded as given");
        assert_eq!(
            src1[..ne],
            built.ids_src1[..ne],
            "the derived inverse is the helper's"
        );
        assert!(
            dst[ne..].iter().all(|&v| v == SENT),
            "ids_dst pad and canary"
        );
        assert!(src1[ne..].iter().all(|&v| v == SENT), "ids_src1 canary");
        assert!(bounds[e + 1..].iter().all(|&v| v == SENT), "bounds canary");
    }
}

#[test]
fn mmid_ids_from_host_refuses_before_any_device_write() {
    if !testkit::device_present("mmid_ids_from_host_refuses_before_any_device_write") {
        return;
    }
    let shape = small();
    let good_dst = [0, 1, 2, 3, 4, 5, 6, 7];
    let good_bounds = [0, 2, 4, 6, 8];
    #[rustfmt::skip]
    let rows: [HostRow<'_>; 5] = [
        ("a duplicated entry", &[0, 1, 2, 3, 4, 5, 6, 6], &good_bounds, false, "ids values: ids_dst[7] repeats"),
        ("an entry equal to m * top_k", &[0, 1, 2, 3, 4, 5, 6, 8], &good_bounds, false, "ids values: ids_dst[7] is 8"),
        ("an expert with m + 1 columns", &good_dst, &[0, 5, 6, 7, 8], false, "ids values: expert 0 holds 5"),
        ("ids_dst one int short of M2", &good_dst, &good_bounds, true, "the CUDA ids_dst range"),
        ("a shape outside the rules", &good_dst, &good_bounds, false, "mmq_mmid: top_k"),
    ];
    let phase = || -> Result<Vec<HostOut>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        rows.iter()
            .map(|&(name, ids_dst, expert_bounds, short_dst, _)| {
                let host = HostIds {
                    ids_dst,
                    expert_bounds,
                };
                let call_shape = match name {
                    "a shape outside the rules" => MmidShape { top_k: 5, ..shape },
                    _ => shape,
                };
                run_from_host(&ctx, &stream, call_shape, host, short_dst)
            })
            .collect()
    };
    let runs = phase().expect("the refusal phase");
    for ((name, _, _, _, needle), (made, back)) in rows.iter().zip(runs) {
        match made {
            Err(EngError::BadRequest(text)) => assert!(text.contains(needle), "{name}: {text}"),
            other => panic!("{name} was not refused: {other:?}"),
        }
        for (range, ints) in ["ids_src1", "ids_dst", "expert_bounds"].iter().zip(back) {
            assert!(
                ints.iter().all(|&v| v == SENT),
                "{name}: {range} was written"
            );
        }
    }
}

/// The Qwen expert gate launch at `m` tokens: Q4_K, K 2,048, N 512 per expert, 256 experts,
/// top 8, the stack's own stride, `ld` 1,024 (gate at column 0 of `gu`).
fn qwen_gate(m: usize) -> MmidLaunchShape {
    MmidLaunchShape {
        ids: qwen(m),
        quant: QuantType::Q4K,
        k: 2048,
        n: 512,
        stride_bytes: 512 * 8 * 144,
        ld: 1024,
    }
}

/// The geometry every grouped launch shares, with the per-shape fields set by the caller.
fn grouped(type_x: i64, k: i64, n: i64, ld: i64) -> MmidGeometry {
    MmidGeometry {
        type_x,
        ncols_x: k,
        nrows_x: n,
        ncols_dst: 0,
        stride_row_x: k / 256,
        ncols_y: 0,
        nrows_dst: ld,
        nchannels_x: 0,
        nchannels_y: 0,
        stride_channel_x: 0,
        stride_channel_y: 0,
        stride_channel_dst: 0,
        nsamples_x: 1,
        nsamples_y: 1,
        stride_sample_x: 0,
        stride_sample_y: 0,
        stride_sample_dst: 0,
        ncols_max: 0,
        ncols_opt: 0,
    }
}

#[test]
fn fill_mmid_args_pins_every_field_of_the_grouped_form() {
    // The pin's assignment (mmq.cu:255-261): ncols_max = ncols_opt = m (49, not 392),
    // ncols_dst = ncols_y = m * top_k, the expert as the channel, the stride in weight blocks,
    // one sample, zero channel and sample strides for y and dst.
    let gate = qwen_gate(49);
    check_launch_shape(&gate).expect("the Qwen gate shape");
    let want = MmidGeometry {
        ncols_dst: 392,
        ncols_y: 392,
        nchannels_x: 256,
        nchannels_y: 256,
        stride_channel_x: 4096,
        ncols_max: 49,
        ncols_opt: 49,
        ..grouped(12, 2048, 512, 1024)
    };
    assert_eq!(fill_mmid_args(&gate), want);
    // The down launch over an 8-slot Q6_K bank (seam I2) whose slots sit 4,800 blocks apart.
    let down = MmidLaunchShape {
        ids: MmidShape {
            n_experts: 8,
            m: 64,
            top_k: 8,
        },
        quant: QuantType::Q6K,
        k: 512,
        n: 2048,
        stride_bytes: 4800 * 210,
        ld: 2048,
    };
    check_launch_shape(&down).expect("the bank down shape");
    let want = MmidGeometry {
        ncols_dst: 512,
        ncols_y: 512,
        nchannels_x: 8,
        nchannels_y: 8,
        stride_channel_x: 4800,
        ncols_max: 64,
        ncols_opt: 64,
        ..grouped(14, 512, 2048, 2048)
    };
    assert_eq!(fill_mmid_args(&down), want);
}

#[test]
fn check_launch_shape_refuses_every_number_rule() {
    let base = qwen_gate(49);
    let with = |edit: &dyn Fn(&mut MmidLaunchShape)| {
        let mut shape = base;
        edit(&mut shape);
        shape
    };
    // 2,048 tokens over 32,768 rows of K 32,768: 2^24 tiles of 128 blocks at J = 8.
    let wide = |n_experts: usize| MmidLaunchShape {
        ids: MmidShape {
            n_experts,
            m: 2048,
            top_k: 8,
        },
        quant: QuantType::Q4K,
        k: 32_768,
        n: 32_768,
        stride_bytes: 32_768 * 128 * 144,
        ld: 32_768,
    };
    // The verdict MINOR-1 witness: 358,416,384 tiles of one block reach i32::MAX when the
    // launcher scales them by 100, while staying under the 2^30 stream-k limit.
    let overflow = MmidLaunchShape {
        ids: MmidShape {
            n_experts: 256,
            m: 2048,
            top_k: 1,
        },
        quant: QuantType::Q4K,
        k: 256,
        n: 700_032,
        stride_bytes: 700_032 * 144,
        ld: 700_032,
    };
    // The tile-count boundary at one k block and 256 experts: 256 token tiles of 8 columns,
    // `row_tiles` tiles of 128 rows. 327 row tiles are 21,430,272 tiles and stay inside the
    // limit of 21,474,836; 328 are 21,495,808 and do not. Both stay under the 2^30 clause.
    let tiles = |row_tiles: usize| MmidLaunchShape {
        ids: MmidShape {
            n_experts: 256,
            m: 2048,
            top_k: 1,
        },
        quant: QuantType::Q4K,
        k: 256,
        n: 128 * row_tiles,
        stride_bytes: 128 * row_tiles * 144,
        ld: 128 * row_tiles,
    };
    let int_max = 0x7fff_ffff_usize;
    let rows: Vec<(&str, MmidLaunchShape)> = vec![
        ("m", with(&|s| s.ids.m = 0)),
        ("type", with(&|s| s.quant = QuantType::Q80)),
        ("type", with(&|s| s.quant = QuantType::Q50)),
        ("K-grid", with(&|s| s.k = 384)),
        ("K-grid", with(&|s| (s.quant, s.k) = (QuantType::Q6K, 384))),
        ("K", with(&|s| s.k = 0)),
        ("K", with(&|s| s.k = 32_768 + 256)),
        ("N", with(&|s| s.n = 0)),
        ("N", with(&|s| (s.n, s.ld) = (int_max + 1, int_max + 1))),
        ("ld", with(&|s| s.ld = 511)),
        ("ld", with(&|s| s.ld = int_max + 1)),
        ("grid limit", wide(256)),
        ("grid limit", overflow),
        ("grid limit", tiles(328)),
        ("stride", with(&|s| s.stride_bytes += 1)),
        ("stride", with(&|s| s.stride_bytes -= 144)),
        ("stride", with(&|s| s.stride_bytes = (int_max + 1) * 144)),
        ("weight index", with(&|s| s.stride_bytes = (1 << 24) * 144)),
        ("dst index", with(&|s| s.ld = int_max)),
    ];
    for (tag, shape) in &rows {
        match check_launch_shape(shape) {
            Err(EngError::BadRequest(text)) => {
                assert!(
                    text.starts_with(&format!("mmq_mmid: {tag}: ")),
                    "{shape:?}: {text}"
                );
            }
            other => panic!("{shape:?} was not refused by {tag}: {other:?}"),
        }
    }
    // The other side of each boundary: the grid just under 2^30 (63 experts), the tile count
    // just under i32::MAX / 100 (327 row tiles over 256 experts), K at both ends of its range,
    // one weight row.
    let inside = [
        base,
        wide(63),
        tiles(327),
        with(&|s| (s.k, s.stride_bytes) = (256, 512 * 144)),
        with(&|s| (s.k, s.stride_bytes) = (32_768, 512 * 128 * 144)),
        with(&|s| (s.n, s.ld, s.stride_bytes) = (1, 1, 8 * 144)),
    ];
    for shape in &inside {
        assert!(
            check_launch_shape(shape).is_ok(),
            "{shape:?} is inside the rules"
        );
    }
}

/// Launch slot order: weight, y, ids_src1, ids_dst, expert_bounds, dst, fixup.
type LaunchSlots = [Slot; 7];

/// What the launch refusal phase gives: each row's name and needle, each row's plan, the
/// base plan.
type LaunchPhase = (Vec<(&'static str, &'static str)>, Vec<Planned>, Planned);

/// The small launch of the refusal rows: 4 experts of 128 rows of K 256 (Q4_K), 4 tokens of 2
/// picks, `ld` 128.
fn small_launch() -> MmidLaunchShape {
    MmidLaunchShape {
        ids: small(),
        quant: QuantType::Q4K,
        k: 256,
        n: 128,
        stride_bytes: 128 * 144,
        ld: 128,
    }
}

/// The base slots of [`small_launch`], each range in its own buffer and sized exactly.
fn launch_slots(fixup_bytes: usize) -> LaunchSlots {
    [
        (0, 0, 512 * 144),
        (1, 0, 2 * 8 * 144 + 18_432),
        (2, 0, 8 * 4),
        (3, 0, (8 + 4 + 127) * 4),
        (4, 0, 5 * 4),
        (5, 0, (7 * 128 + 128) * 4),
        (6, 0, fixup_bytes),
    ]
}

/// Plans the launch the slots describe (no launch), with an `MmidIds` built literally over the
/// id slots for `ids_shape`.
fn plan_launch(
    ctx: &CudaContext,
    bufs: &[DeviceBuffer],
    launch: &MmidLaunchShape,
    slots: &LaunchSlots,
    ids_shape: MmidShape,
) -> Result<(), EngError> {
    let r = |i: usize| bufs[slots[i].0].range(slots[i].1, slots[i].2);
    let ids = MmidIds {
        out: MmidIdsOut {
            ids_src1: r(2)?,
            ids_dst: r(3)?,
            expert_bounds: r(4)?,
        },
        shape: ids_shape,
    };
    let call = MmidLaunchCall {
        shape: launch.ids,
        weight: MmidWeight {
            base: r(0)?,
            quant: launch.quant,
            k: launch.k,
            n: launch.n,
            stride_bytes: launch.stride_bytes,
        },
        ids: &ids,
        y: r(1)?,
        dst: r(5)?,
        ld: launch.ld,
        fixup: r(6)?,
    };
    ctx.plan_mmid_launch(&call).map(|_| ())
}

#[test]
fn mmq_mmid_launch_refuses_ranges_that_do_not_fit() {
    if !testkit::device_present("mmq_mmid_launch_refuses_ranges_that_do_not_fit") {
        return;
    }
    let launch = small_launch();
    // Each row: a name, slot edits, the ids' shape, the launch numbers, a needle of the text.
    type LaunchRow = (
        &'static str,
        Vec<(usize, Slot)>,
        MmidShape,
        MmidLaunchShape,
        &'static str,
    );
    let phase = || -> Result<LaunchPhase, EngError> {
        let ctx = CudaContext::new()?;
        let fixup = 4 * 128 * 128 * device_facts(&ctx)?.nsm;
        let base = launch_slots(fixup);
        let ids = small();
        let k384 = MmidLaunchShape { k: 384, ..launch };
        #[rustfmt::skip]
        let rows: Vec<LaunchRow> = vec![
            ("weight one block short", vec![(0, (0, 0, 512 * 144 - 144))], ids, launch, "the CUDA weight range"),
            ("weight misaligned", vec![(0, (0, 4, 512 * 144))], ids, launch, "weight range starts at offset 4"),
            ("y one byte short of rule M1", vec![(1, (1, 0, 20_735))], ids, launch, "the CUDA y range"),
            ("y misaligned", vec![(1, (1, 4, 20_736))], ids, launch, "y range starts at offset 4"),
            ("ids_dst one int short of rule M2", vec![(3, (3, 0, 138 * 4))], ids, launch, "the CUDA ids_dst range"),
            ("ids_dst at the rev 2 form", vec![(3, (3, 0, 135 * 4))], ids, launch, "the CUDA ids_dst range"),
            ("expert_bounds n_experts ints", vec![(4, (4, 0, 16))], ids, launch, "the CUDA expert_bounds range"),
            ("dst one float short", vec![(5, (5, 0, 4092))], ids, launch, "the CUDA dst range"),
            ("dst misaligned", vec![(5, (5, 2, 4096))], ids, launch, "dst range starts at offset 2"),
            ("fixup one float short", vec![(6, (6, 0, fixup - 4))], ids, launch, "the CUDA fixup range"),
            ("fixup misaligned", vec![(6, (6, 4, fixup))], ids, launch, "fixup range starts at offset 4"),
            ("dst over y", vec![(5, (1, 0, 4096))], ids, launch, "the CUDA dst and y ranges overlap"),
            ("dst over ids_dst", vec![(5, (3, 0, 4096))], ids, launch, "the CUDA dst and ids_dst ranges overlap"),
            ("dst over fixup", vec![(5, (6, 0, 4096))], ids, launch, "the CUDA dst and fixup ranges overlap"),
            ("weight under fixup", vec![(0, (6, 0, 512 * 144))], ids, launch, "the CUDA fixup and weight ranges overlap"),
            ("ids made for another shape", vec![], MmidShape { m: 3, ..ids }, launch, "mmq_mmid: ids shape"),
            ("K off the grid", vec![], ids, k384, "mmq_mmid: K-grid"),
        ];
        let bufs: Vec<DeviceBuffer> = base
            .iter()
            .map(|&(_, _, bytes)| ctx.upload(&vec![0u8; 2 * bytes + SLACK]))
            .collect::<Result<_, _>>()?;
        let results = rows
            .iter()
            .map(|(_, edits, ids_shape, numbers, _)| {
                let mut slots = base;
                for &(i, slot) in edits {
                    slots[i] = slot;
                }
                plan_launch(&ctx, &bufs, numbers, &slots, *ids_shape)
            })
            .collect();
        let base_ok = plan_launch(&ctx, &bufs, &launch, &base, ids);
        let names = rows.iter().map(|row| (row.0, row.4)).collect();
        Ok((names, results, base_ok))
    };
    let (names, results, base_ok) = phase().expect("the refusal phase");
    assert!(base_ok.is_ok(), "the base launch plans: {base_ok:?}");
    for ((name, needle), result) in names.iter().zip(results) {
        match result {
            Err(EngError::BadRequest(text)) => assert!(text.contains(needle), "{name}: {text}"),
            other => panic!("{name} was not refused: {other:?}"),
        }
    }
}

#[test]
fn the_grouped_tile_hint_is_clamped_to_a_tiling_width() {
    // auto for 128 tokens x top_k 8 over 256 experts is four columns, which the chooser rounds to
    // its smallest tile (J=8).
    assert_eq!(hint_cols(1024, 256), 4);
    // The clamp keeps the grouped launch off the stream-k configs, which start above 64.
    assert_eq!(clamp_hint(4, 128), 4);
    assert_eq!(clamp_hint(80, 128), GROUPED_TILE_MAX);
    assert_eq!(clamp_hint(1, 128), 1);
    assert_eq!(clamp_hint(0, 128), 1);
    // Never past the token count, and never below one column.
    assert_eq!(clamp_hint(64, 32), 32);
    assert_eq!(clamp_hint(i64::MAX, 1), 1);
    assert_eq!(hint_cols(0, 0), 1);
    assert_eq!(hint_cols(usize::MAX, 1), i64::MAX);
}

#[test]
fn the_tile_hint_reaches_q4_k_only() {
    // Q6_K has no tiling row in the vendored table, so its launches stay stream-k; a narrower
    // tile there would change the per-column reduction order without pg1b's guarantee.
    let shape = |quant| MmidLaunchShape {
        ids: MmidShape {
            n_experts: 256,
            m: 128,
            top_k: 8,
        },
        quant,
        k: 2048,
        n: 512,
        stride_bytes: 0,
        ld: 512,
    };
    set_hint_override(Some(Some(8)));
    let q4k = mmid_geometry(&shape(QuantType::Q4K));
    let q6k = mmid_geometry(&shape(QuantType::Q6K));
    set_hint_override(None);
    assert_eq!(q4k.ncols_opt, 8, "Q4_K follows the hint");
    assert_eq!(q6k.ncols_opt, 128, "Q6_K keeps the whole-chunk tile");
}
