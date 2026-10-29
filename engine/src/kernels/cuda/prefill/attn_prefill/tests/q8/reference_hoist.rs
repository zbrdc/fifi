//! Bitwise old-vs-new fixture for the q8 group-scalar hoist: the hoisted kernel and the
//! pre-hoist per-element math are two template instantiations in one ELF, run on identical
//! device inputs, and every output byte (out, part, guards, slack) must be equal. Determinism
//! is the property under test: same accumulation order, only the scalar loads move. Full-attn
//! mode only; the segmented mode shares the same kernel body and stays covered by the numerical
//! tests.

use super::*;

/// One run's snapshot: out bytes, part bytes, status word, whole status buffer.
type RunSnapshot = (Vec<u8>, Vec<u8>, i32, Vec<u8>);

/// Enrich a plane's scales with the 7/3/5 hash: q8.rs's plane() keeps its 4-value hash for the
/// ungated oracle test, so the fixture rewrites scale bytes here (and the mirrored f64 values,
/// signed-aware like spike_v) for the wider per-row variety the tolerance evidence wants.
fn enrich_scales(p: &mut Plane, shape: AttnPrefillShape, signed: bool, role: usize) {
    let AttnKvStorage::Q8Affine { group_size, .. } = p.layout.storage else {
        panic!("test storage")
    };
    let Some(scales) = p.layout.scales else {
        return;
    };
    for row in 0..shape.cache_rows {
        for h in 0..shape.n_kv_head {
            for g in 0..shape.head_dim / group_size {
                let scale = (1 + (7 * row + 3 * h + 5 * g + role) % 11) as f32 / 128.0;
                let offset = ORIGINS[1]
                    + row * scales.strides.time_stride_bytes
                    + h * scales.strides.head_stride_bytes
                    + g * scales.group_stride_bytes;
                put_scalar(&mut p.bytes[1], offset, scales.dtype, scale);
                for lane in 0..group_size {
                    let d = g * group_size + lane;
                    let code = p.bytes[0][ORIGINS[0]
                        + row * p.layout.payload_strides.time_stride_bytes
                        + h * p.layout.payload_strides.head_stride_bytes
                        + d];
                    let code = if signed {
                        f64::from(code as i8)
                    } else {
                        f64::from(code)
                    };
                    p.values[(row * shape.n_kv_head + h) * shape.head_dim + d] =
                        scale as f64 * code;
                }
            }
        }
    }
}

/// Plant whole-row V code spikes: window-boundary errors that leak an invisible key, or miss a
/// visible one, move the output by whole units instead of rounding noise.
fn spike_v(v: &mut Plane, shape: AttnPrefillShape, rows: &[usize], signed: bool) {
    let AttnKvStorage::Q8Affine { group_size, .. } = v.layout.storage else {
        panic!("test storage")
    };
    for &row in rows {
        for h in 0..shape.n_kv_head {
            for d in 0..shape.head_dim {
                let offset = ORIGINS[0]
                    + row * v.layout.payload_strides.time_stride_bytes
                    + h * v.layout.payload_strides.head_stride_bytes
                    + d;
                let code: u8 = if d % 2 == 0 {
                    127
                } else if signed {
                    129
                } else {
                    255
                };
                v.bytes[0][offset] = code;
                let code = if signed {
                    f64::from(code as i8)
                } else {
                    f64::from(code)
                };
                let scale = f64::from(
                    (1 + (7 * row + 3 * h + 5 * (d / group_size) + 1) % 11) as f32 / 128.0,
                );
                v.values[(row * shape.n_kv_head + h) * shape.head_dim + d] = scale * code;
            }
        }
    }
}

/// The Rust mirror of the C enqueue predicate (attn_prefill.cu enqueue_q8): the tiled
/// kernel serves full-span D128 launches with both planes in 32-dim groups and G <= 8.
/// One definition for the launch choice, the comparison class and the test filters.
fn routes_tiled(shape: &AttnPrefillShape, group_size: usize, segmented: bool) -> bool {
    shape.head_dim == 128 && group_size == 32 && !segmented && shape.n_head / shape.n_kv_head <= 8
}

fn hoist_case(
    case: Case,
    min_keys: usize,
    v_spikes: &[usize],
    use_tiled: bool,
    q_overflow: bool,
) -> Result<Vec<RunSnapshot>, EngError> {
    let shape = case.shape;
    let storage = AttnKvStorage::Q8Affine {
        group_size: case.group_size,
        signed: case.signed,
    };
    let mut k = plane(shape, storage, case.scalar, 0);
    let mut v = plane(shape, storage, case.scalar, 1);
    // The 7/3/5 scale hash lives here (q8.rs's plane keeps its 4-value hash for the ungated
    // oracle test).
    enrich_scales(&mut k, shape, case.signed, 0);
    enrich_scales(&mut v, shape, case.signed, 1);
    if !v_spikes.is_empty() {
        spike_v(&mut v, shape, v_spikes, case.signed);
    }
    let q_len = shape.m * shape.n_head * shape.head_dim;
    let kv_len = shape.cache_rows * shape.n_kv_head * shape.head_dim;
    let input = Data {
        // Regression lever: finite q at -FLT_MAX overflows the dot product to -inf, so a
        // visible key's score is non-finite and the row must poison (NaN).
        q: if q_overflow {
            vec![-f32::MAX; q_len]
        } else {
            (0..q_len)
                .map(|i| ((i % 31) as f32 - 15.0) / 256.0)
                .collect()
        },
        k: vec![0; kv_len],
        v: vec![0; kv_len],
        proj: vec![0.25; shape.m * shape.ld],
    };
    let split = CpfAttnSplit {
        max_splits: case.splits,
        min_keys,
    };
    let mut fx = Fx::with_data(shape, split, case.gate, true, &input)?;
    fx.span = case.span;
    let [kb, vb] = [k.upload(&fx.ctx)?, v.upload(&fx.ctx)?];
    let views = [k.view(&kb)?, v.view(&vb)?];
    let sinks: Vec<f32> = (0..shape.n_head).map(|h| h as f32 / 16.0 - 1.0).collect();
    if case.sinks {
        fx.sink_buf = Some(fx.ctx.upload(&le(&sinks))?);
    }
    let call = || -> Result<AttnPrefillCall<'_>, EngError> {
        let mut call = fx.call()?;
        call.storage = storage;
        call.k_cache = views[0].payload;
        call.v_cache = views[1].payload;
        call.kv_views = Some(views);
        Ok(call)
    };
    let mut snapshots = Vec::new();
    // The tiled kernel is reachable only through the fixture entry — and only when the CALLER
    // asks for it (the hoist bit-identity test keeps the old kernel). The shape guards still
    // apply inside the launch predicate.
    let tiled = use_tiled && routes_tiled(&shape, case.group_size, case.segmented);
    for reference in [false, false, true] {
        fx.reset_outputs()?;
        if reference {
            fx.ctx.attn_prefill_q8_reference(&fx.stream, call()?)?;
        } else if tiled {
            fx.ctx.attn_prefill_q8_tiled(&fx.stream, call()?)?;
        } else {
            fx.ctx.attn_prefill(&fx.stream, call()?)?;
        }
        fx.stream.synchronize()?;
        // The PART slack tail must stay pristine on the tiled branch too — a tiled write past
        // the combine's real extent would land here.
        assert!(
            fx.bytes(PART)[fx.base[PART].2..]
                .iter()
                .all(|&b| b == PART_FILL),
            "part slack tail mutated by a {} run",
            if tiled { "tiled" } else { "old" }
        );
        snapshots.push((
            fx.bytes(OUT),
            fx.bytes(PART),
            fx.status_word(),
            fx.bytes(STATUS),
        ));
    }
    Ok(snapshots)
}

#[test]
fn q8_group_scalar_hoist_is_bit_identical_to_per_element_math() {
    if !testkit::device_present("q8_group_scalar_hoist_is_bit_identical_to_per_element_math") {
        return;
    }
    let flint = |m: usize, p0: i64, span: CpfAttnSpan, splits: usize| Case {
        shape: AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            cache_rows: if span.ring > 0 { 2048 } else { 3200 },
            ld: 0,
            p0,
        },
        signed: true,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span,
        splits,
        sinks: true,
        gate: false,
        segmented: false,
    };
    let list = || -> Vec<Case> {
        vec![
            // Production prefill shape: one row at depth 3200, full attention.
            flint(1, 3199, CpfAttnSpan::FULL, 1),
            // A whole C8 chunk at depth, and the same through the split path.
            flint(8, 3192, CpfAttnSpan::FULL, 4),
            // Sliding window 513 at depth (the boundary case of the depth-513 ring form).
            flint(
                8,
                3192,
                CpfAttnSpan {
                    window: 513,
                    ring: 0,
                },
                2,
            ),
            // Ring wrap: absolute positions straddle a ring multiple, physical rows rotate.
            flint(
                8,
                4100,
                CpfAttnSpan {
                    window: 513,
                    ring: 2048,
                },
                2,
            ),
            // Padding warps (D64 runs 128 threads) and four groups per warp lane span (D256).
            Case {
                shape: AttnPrefillShape {
                    m: 17,
                    n_head: 4,
                    n_kv_head: 2,
                    head_dim: 64,
                    cache_rows: 65,
                    // The gate path needs a real proj: ld = n_head * 2 * head_dim (check_span_split).
                    ld: 4 * 2 * 64,
                    p0: 48,
                },
                signed: true,
                group_size: 32,
                scalar: (Dtype::F16, None),
                span: CpfAttnSpan {
                    window: 19,
                    ring: 0,
                },
                splits: 4,
                sinks: false,
                gate: true,
                segmented: false,
            },
            Case {
                shape: AttnPrefillShape {
                    m: 17,
                    n_head: 6,
                    n_kv_head: 2,
                    head_dim: 256,
                    cache_rows: 65,
                    ld: 0,
                    p0: 48,
                },
                signed: false,
                group_size: 32,
                scalar: (Dtype::F16, Some(Dtype::F16)),
                span: CpfAttnSpan::FULL,
                splits: 4,
                sinks: true,
                gate: false,
                segmented: false,
            },
            // Control: group_size != 32 keeps both instantiations on the per-element path.
            Case {
                shape: AttnPrefillShape {
                    m: 17,
                    n_head: 4,
                    n_kv_head: 2,
                    head_dim: 128,
                    cache_rows: 65,
                    ld: 0,
                    p0: 48,
                },
                signed: true,
                group_size: 64,
                scalar: (Dtype::F16, None),
                span: CpfAttnSpan::FULL,
                splits: 4,
                sinks: false,
                gate: false,
                segmented: false,
            },
            // A D128 unsigned+bias ROUTING case (the only other unsigned+bias arm is D256,
            // which the predicate never routes tiled).
            Case {
                shape: AttnPrefillShape {
                    m: 8,
                    n_head: 32,
                    n_kv_head: 8,
                    head_dim: 128,
                    cache_rows: 3200,
                    ld: 0,
                    p0: 3192,
                },
                signed: false,
                group_size: 32,
                scalar: (Dtype::F16, Some(Dtype::F16)),
                span: CpfAttnSpan::FULL,
                splits: 2,
                sinks: false,
                gate: false,
                segmented: false,
            },
        ]
    };
    // EVERY hoist case — ring wrap, sliding, sinks, the D64 gate arm, unsigned codes, D256
    // bias — also runs the tiled routing (check_hoist_case re-classifies: routing shapes get
    // tolerance + determinism, the non-routing controls — segmented, group 64, D256 — stay
    // bitwise on the old kernel).
    for case in list() {
        check_hoist_case(case, 32, &[], false);
    }
    for case in list() {
        check_hoist_case(case, 32, &[], true);
    }
}

/// One classification + assertion pass over a case's three runs (tiled, tiled repeat,
/// reference). Shapes the enqueue predicate routes to attn_prefill_part_q8_tile_k reorder the
/// summation (per-tile softmax), so old-vs-new is a stated tolerance (2^-14 * max|out_row|
/// against the reference arm) plus a two-run determinism check; everything else keeps the
/// bitwise contract.
fn check_hoist_case(case: Case, min_keys: usize, v_spikes: &[usize], use_tiled: bool) {
    let shape = case.shape;
    let tiled = use_tiled && routes_tiled(&shape, case.group_size, case.segmented);
    let mut result = hoist_case(case, min_keys, v_spikes, use_tiled, false)
        .expect("q8 hoist fixture device phase");
    let (out_r, part_r, status_r, status_bytes_r) = {
        let (o, p, s, sb) = result.pop().expect("reference run");
        (o, p, s, sb)
    };
    // Index the two non-reference runs explicitly: a pop-then-last order would compare the
    // SECOND snapshot with itself and make the determinism assert vacuous. Run 1's whole
    // snapshot (out + part + status bytes) is compared — discarding the part bytes would hide
    // combine-input drift.
    let (out_h, part_h, status_h, status_bytes_h) = result.pop().expect("second run");
    let (out_n2, part_n2, _, status_bytes_n2) = result.pop().expect("first run");
    let label = format!(
        "D{}/GQA{}/m{}/window{}/ring{}/group{}",
        shape.head_dim,
        shape.n_head / shape.n_kv_head,
        shape.m,
        case.span.window,
        case.span.ring,
        case.group_size,
    );
    assert_eq!(status_h, 0, "hoisted status {label}");
    assert_eq!(status_r, 0, "reference status {label}");
    assert_eq!(status_bytes_h, status_bytes_r, "status bytes {label}");
    // The OUT buffer carries a 0x55 slack tail whose bytes read as ~1.47e13 f32 — a bound
    // folded over the WHOLE buffer is ~9e8 and only NaN can fail it. The slack must stay
    // pristine (guard bytes) and the tolerance only sees real rows.
    let out_bytes = shape.m * shape.n_head * shape.head_dim * 4;
    for (name, out) in [
        ("first", &out_n2),
        ("second", &out_h),
        ("reference", &out_r),
    ] {
        assert!(
            out[out_bytes..].iter().all(|&b| b == 0x55),
            "slack tail mutated by {name} run {label}"
        );
    }
    if tiled {
        assert!(
            (
                out_h.as_slice(),
                part_h.as_slice(),
                status_bytes_h.as_slice()
            ) == (
                out_n2.as_slice(),
                part_n2.as_slice(),
                status_bytes_n2.as_slice(),
            ),
            "determinism: two tiled runs {label}"
        );
        // Per-row bound: each row's tolerance is scaled by that row's own maximum magnitude, so
        // one huge row cannot mask another's error.
        let rows = shape.m * shape.n_head;
        fn row_window(b: &[u8], row: usize, head_dim: usize) -> &[u8] {
            &b[row * head_dim * 4..(row + 1) * head_dim * 4]
        }
        // The 2^-14 bound is empirical — print the observed worst |diff|/bound and the smallest
        // bound so a near-miss row is visible, not just pass/fail (low-magnitude m1 rows are
        // the suspected tight case).
        let (mut worst, mut worst_row, mut min_bound) = (0.0f32, 0usize, f32::INFINITY);
        for row in 0..rows {
            let (hr, rr) = (
                row_window(&out_h, row, shape.head_dim),
                row_window(&out_r, row, shape.head_dim),
            );
            let bound = rr
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .fold(0.0f32, |m, x| m.max(x.abs()))
                * 2.0f32.powi(-14);
            min_bound = min_bound.min(bound);
            for (h, r) in hr
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rr.as_chunks::<4>().0.iter())
            {
                let d = (f32::from_le_bytes(*h) - f32::from_le_bytes(*r)).abs();
                // Finiteness: a NaN on either side must fail HERE, not inside the bound compare
                // where NaN <= bound reads as a confusing near-miss.
                assert!(
                    f32::from_le_bytes(*h).is_finite() && f32::from_le_bytes(*r).is_finite(),
                    "non-finite output float in row {row} {label}"
                );
                if bound > 0.0 {
                    let ratio = d / bound;
                    if ratio > worst {
                        worst = ratio;
                        worst_row = row;
                    }
                }
                assert!(
                    d <= bound,
                    "tiled-vs-old row {row} {label}: |{d}| > {bound}"
                );
            }
        }
        eprintln!(
            "[q8-tiled-margin] {label}: worst |diff|/bound = {worst:.4} at row {worst_row}, min bound {min_bound:.3e} (2^-14 per-row, report-only)"
        );
    } else {
        assert_eq!(out_h, out_r, "out bytes {label}");
        // The PART snapshot comparison is part of the old-kernel contract — the combine
        // consumes these bytes, so they must match byte for byte too.
        assert_eq!(part_h, part_r, "part bytes {label}");
    }
}

/// Tiled-kernel sensitivity cases: the production GQA (32/4, G=8 — a different warp-to-head
/// mapping than the hoist cases' G=4) at the m=1 partial tile, and V spikes planted at the
/// window boundary rows p-513/p-512 where an off-by-one visibility rule must fail loudly
/// instead of inside rounding noise. Ring wrap, m8, sliding and sinks ride the hoist test's
/// tiled second pass above; the arm below adds the production-plan geometry itself (41 splits,
/// min_keys 256 — cpf_attn_launch_plan's m1/KVH4 answer, pinned by the host planner test).
#[test]
fn q8_tiled_kernel_old_vs_new_tolerance_with_window_spikes() {
    if !testkit::device_present("q8_tiled_kernel_old_vs_new_tolerance_with_window_spikes") {
        return;
    }
    let flint = |m: usize, p0: i64, span: CpfAttnSpan, splits: usize| Case {
        shape: AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            cache_rows: 3200,
            ld: 0,
            p0,
        },
        signed: true,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span,
        splits,
        sinks: true,
        gate: false,
        segmented: false,
    };
    // last = p0 + m - 1 = 3199: rows 2686 (=p-513, invisible) and 2687 (=p-512, first visible)
    // carry whole-row spikes. The m8 GQA8 and m17 GQA8 arms are the cross-token shapes: odd
    // tiles per split with padding warps that must still reach the per-token barrier; m8 GQA4
    // covers the four-active-warp grid.
    let window513 = CpfAttnSpan {
        window: 513,
        ring: 0,
    };
    let g4 = |m: usize, p0: i64, span: CpfAttnSpan, splits: usize| Case {
        shape: AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            cache_rows: 3200,
            ld: 0,
            p0,
        },
        signed: true,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span,
        splits,
        sinks: true,
        gate: false,
        segmented: false,
    };
    for (case, spikes) in [
        (flint(1, 3199, CpfAttnSpan::FULL, 1), &[][..]),
        (flint(8, 3192, window513, 2), &[2686, 2687][..]),
        (g4(8, 3192, window513, 2), &[2686, 2687][..]),
        (flint(17, 3183, window513, 3), &[2686, 2687][..]),
        // The production plan itself — 41 splits at min_keys 256, the geometry the
        // microbench's Q8Hoisted arm measures.
        (flint(1, 3199, CpfAttnSpan::FULL, 41), &[][..]),
    ] {
        let keys = if case.splits == 41 { 256 } else { 32 };
        check_hoist_case(case, keys, spikes, true);
    }
}

/// Regression: a visible key whose score overflows to -inf on FINITE inputs (q = -FLT_MAX) must
/// poison the row to NaN exactly like the old kernel — the tiled path once returned a silent 0
/// weight instead, which reads as a clean (wrong) answer downstream. UNSIGNED codes and no
/// sinks/bias keep every dot term a clean -inf (signed codes give mixed-sign lane partials, NaN
/// in the shuffle reduction, which the old -inf sentinel skip would ALSO have poisoned — no
/// regression power).
#[test]
fn q8_tiled_visible_overflow_poisons_like_the_old_kernel() {
    if !testkit::device_present("q8_tiled_visible_overflow_poisons_like_the_old_kernel") {
        return;
    }
    let case = Case {
        shape: AttnPrefillShape {
            m: 1,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            cache_rows: 2,
            ld: 0,
            p0: 1,
        },
        signed: false,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span: CpfAttnSpan::FULL,
        splits: 1,
        sinks: false,
        gate: false,
        segmented: false,
    };
    // min_keys must satisfy the wrapper contract (multiple of 32, >= 32) — min_keys 2 errors in
    // the wrapper before any kernel runs.
    let out_bytes = case.shape.m * case.shape.n_head * case.shape.head_dim * 4;
    for use_tiled in [true, false] {
        let mut runs =
            hoist_case(case, 32, &[], use_tiled, true).expect("q8 overflow fixture device phase");
        // Determinism first: both non-reference runs identical, whole snapshot.
        let (out_r, _, status_r, status_bytes_r) = runs.pop().expect("reference run");
        let second = runs.pop().expect("second run");
        let first = runs.pop().expect("first run");
        assert_eq!(first, second, "determinism (tiled={use_tiled})");
        // Status + OUT slack guards: same contract as the tolerance path.
        assert_eq!(second.2, 0, "status (tiled={use_tiled})");
        assert_eq!(status_r, 0, "reference status (tiled={use_tiled})");
        assert_eq!(second.3, status_bytes_r, "status bytes (tiled={use_tiled})");
        // Then the poison: every real float of every run's output is a NaN — a finite
        // 0 anywhere is exactly the silent-skip bug this pins.
        for (name, out) in [
            ("first", &first.0),
            ("second", &second.0),
            ("reference", &out_r),
        ] {
            assert!(
                out[out_bytes..].iter().all(|&b| b == 0x55),
                "slack tail mutated by {name} run (tiled={use_tiled})"
            );
            let finite = out[..out_bytes]
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|c| !f32::from_le_bytes(**c).is_nan())
                .count();
            assert_eq!(
                finite, 0,
                "{name} run (tiled={use_tiled}) kept {finite} finite floats"
            );
        }
    }
}

/// Report-only microbench: the kernel is ~140x off the bandwidth floor at m=1 (~26 MB unique
/// bytes vs 4.26 ms), so the cost is latency/parallelism, not instruction issue. These arms
/// measure the gaps before any redesign is picked: dtype gap (q8 reference vs the F16 kernel),
/// row batching (m=8 one launch vs 8 x m=1), split parallelism (max_splits=8). Host-timed per
/// launch with explicit sync (the engine has no CUDA-event API; kernels are ~ms). Asserts
/// NOTHING - the design decision belongs to the reviewers.
#[test]
fn q8_prefill_parallelism_microbench_report() {
    use std::time::Instant;
    if !testkit::device_present("q8_prefill_parallelism_microbench_report") {
        return;
    }
    const WARM: usize = 5;
    const N: usize = 20;

    #[derive(Clone, Copy)]
    enum Arm {
        Q8Reference,
        Q8Hoisted,
        F16,
    }
    fn arm_name(a: Arm) -> &'static str {
        match a {
            Arm::Q8Reference => "q8-reference",
            Arm::Q8Hoisted => "q8-hoisted",
            Arm::F16 => "f16",
        }
    }

    // (arm, m, max_splits, min_keys): v3 arms at the production flint shape. The pair
    // (41, 256) is what cpf_attn_launch_plan returns for m1/KVH4 (pinned by the host
    // test); min_keys 128 is the only remaining planner knob. The F16 m8 arm stays
    // report-only: if m-amortisation holds at the production plan, batching a whole
    // 8-row piece into one launch is a separate queued lever for the default F16 path.
    let arms: [(Arm, usize, usize, usize); 6] = [
        (Arm::Q8Reference, 1, 1, 32),
        (Arm::Q8Reference, 1, 41, 256),
        (Arm::Q8Reference, 1, 41, 128),
        // The tiled kernel at the production plan: the enqueue predicate routes this arm to
        // attn_prefill_part_q8_tile_k, so its number IS the <=3x-F16 kernel bar measurement
        // (Q8Reference stays the old-path baseline — the DEFAULT kernel). The report must
        // name the kernel actually selected, and the test asserts the routing.
        (Arm::Q8Hoisted, 1, 41, 256),
        (Arm::F16, 1, 41, 256),
        (Arm::F16, 8, 41, 256),
    ];

    fn stats(mut xs: Vec<u128>) -> String {
        xs.sort_unstable();
        let n = xs.len();
        let mean = xs.iter().sum::<u128>() as f64 / n as f64;
        format!("{} {} {:.0} {}", xs[0], xs[n / 2], mean, xs[n - 1])
    }

    fn bench_arm(arm: Arm, m: usize, max_splits: usize, min_keys: usize) {
        let shape = AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            cache_rows: 3200,
            ld: 0,
            p0: 3200 - m as i64,
        };
        let q8 = !matches!(arm, Arm::F16);
        let storage = AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: true,
        };
        let k = q8.then(|| plane(shape, storage, (Dtype::F16, None), 0));
        let v = q8.then(|| plane(shape, storage, (Dtype::F16, None), 1));
        let input = Data {
            q: vec![1.0 / 256.0; shape.m * shape.n_head * shape.head_dim],
            k: vec![0u16; shape.cache_rows * shape.n_kv_head * shape.head_dim],
            v: vec![0u16; shape.cache_rows * shape.n_kv_head * shape.head_dim],
            proj: Vec::new(),
        };
        let fx = Fx::with_data(
            shape,
            CpfAttnSplit {
                max_splits,
                min_keys,
            },
            false,
            true,
            &input,
        )
        .expect("microbench harness");
        // Owned buffers stay in scope; the views borrow them for the whole arm loop.
        let kb = k.as_ref().map(|p| p.upload(&fx.ctx).expect("k upload"));
        let vb = v.as_ref().map(|p| p.upload(&fx.ctx).expect("v upload"));
        let views = match (k.as_ref(), v.as_ref(), kb.as_ref(), vb.as_ref()) {
            (Some(kp), Some(vp), Some(kb), Some(vb)) => {
                Some([kp.view(kb).expect("k view"), vp.view(vb).expect("v view")])
            }
            _ => None,
        };
        let sinks: Vec<f32> = (0..shape.n_head).map(|h| h as f32 / 16.0 - 1.0).collect();
        let sink_buf = fx.ctx.upload(&le(&sinks)).expect("sinks");
        let call = || -> AttnPrefillCall<'_> {
            let mut call = fx.call().expect("call");
            call.sinks = Some(sink_buf.range(0, shape.n_head * 4).expect("sink range"));
            if q8 {
                call.storage = storage;
                let views = views.as_ref().expect("q8 views");
                call.k_cache = views[0].payload;
                call.v_cache = views[1].payload;
                call.kv_views = Some(*views);
            }
            call
        };
        let run = || {
            if matches!(arm, Arm::Q8Reference) {
                fx.ctx
                    .attn_prefill_q8_reference(&fx.stream, call())
                    .expect("reference launch");
            } else if matches!(arm, Arm::Q8Hoisted) {
                // The tiled kernel is reachable only through the fixture entry; the default
                // launch keeps the old kernel.
                fx.ctx
                    .attn_prefill_q8_tiled(&fx.stream, call())
                    .expect("tiled launch");
            } else {
                fx.ctx.attn_prefill(&fx.stream, call()).expect("launch");
            }
            fx.stream.synchronize().expect("sync");
        };
        for _ in 0..WARM {
            run();
        }
        let mut per_launch = Vec::new();
        for _ in 0..N {
            let t = Instant::now();
            run();
            per_launch.push(t.elapsed().as_micros());
        }
        let t = Instant::now();
        for _ in 0..N {
            if matches!(arm, Arm::Q8Reference) {
                fx.ctx
                    .attn_prefill_q8_reference(&fx.stream, call())
                    .expect("reference launch");
            } else if matches!(arm, Arm::Q8Hoisted) {
                fx.ctx
                    .attn_prefill_q8_tiled(&fx.stream, call())
                    .expect("tiled launch");
            } else {
                fx.ctx.attn_prefill(&fx.stream, call()).expect("launch");
            }
        }
        fx.stream.synchronize().expect("sync");
        let batched = t.elapsed().as_micros();
        // Grid: old path (splits x n_head) blocks of 128 threads (4 warps); tiled path
        // (q_tiles x n_kv_head x splits) blocks of 256 threads (8 warps). The printed
        // warps/SM assumes a 3090's 82 SMs (CPF_ATTN_SM_COUNT is the planner's figure).
        let tiled = matches!(arm, Arm::Q8Hoisted);
        let kernel = match arm {
            Arm::Q8Reference => "attn_prefill_part_q8_k<128>",
            Arm::Q8Hoisted => "attn_prefill_part_q8_tile_k<128>",
            Arm::F16 => "attn_prefill_part_dsplit_k<128,8>",
        };
        // Assert the routing so the label can never lie — the flint bench shape (D128, group
        // 32, GQA 32/4, mode 0) must satisfy the enqueue predicate verbatim.
        assert!(
            !tiled
                || (shape.head_dim == 128
                    && shape.n_head / shape.n_kv_head <= 8
                    && matches!(storage, AttnKvStorage::Q8Affine { group_size: 32, .. })),
            "Q8Hoisted arm no longer routes to the tiled kernel"
        );
        let (blocks, warps_per_block) = if tiled {
            (max_splits * shape.n_kv_head, 8)
        } else {
            (max_splits * shape.n_head, 4)
        };
        eprintln!(
            "Q8_MICROBENCH_V3 {}[kernel={kernel}] m{} splits{}/mk{}: min/med/mean/max [{}] us, batched{} {:.0} us/launch, blocks {} warps/SM<= {:.2}",
            arm_name(arm),
            m,
            max_splits,
            min_keys,
            stats(per_launch),
            N,
            batched as f64 / N as f64,
            blocks,
            (blocks * warps_per_block) as f64 / 82.0,
        );
    }

    // One line per arm or an explicit ARM-REFUSED line; the test fails if any
    // expected arm is missing, so a refused arm can never masquerade as a pass.
    let mut reported: Vec<&str> = Vec::new();
    for (arm, m, max_splits, min_keys) in arms {
        let name = arm_name(arm);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bench_arm(arm, m, max_splits, min_keys);
        }));
        match r {
            Ok(()) => reported.push(name),
            Err(e) => eprintln!(
                "Q8_MICROBENCH_V3 {} m{} splits{}/mk{}: ARM-REFUSED {:?}",
                name,
                m,
                max_splits,
                min_keys,
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| { e.downcast_ref::<&str>().map(|s| s.to_string()) })
            ),
        }
    }
    let expected: Vec<&str> = arms.iter().map(|(a, _, _, _)| arm_name(*a)).collect();
    assert_eq!(reported, expected, "microbench arms missing or refused");
}

/// The production-plan arm: a split-count change reorders f32 partial sums only — both arms
/// read identical quantized bytes — so the bound is pure reduction reorder over the case depth
/// (max_abs 1e-3 on attention outputs of O(30), max_rel 1e-4 against a magnitude floor of 1),
/// plus run-to-run bit-equality at the same plan.
#[test]
fn q8_reference_at_planner_splits_stays_within_reduction_reorder_tolerance() {
    if !testkit::device_present("q8_reference_at_planner_splits") {
        return;
    }
    let flint = |m: usize, p0: i64, span: CpfAttnSpan| Case {
        shape: AttnPrefillShape {
            m,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            cache_rows: if span.ring > 0 { 2048 } else { 3200 },
            ld: 0,
            p0,
        },
        signed: true,
        group_size: 32,
        scalar: (Dtype::F16, None),
        span,
        splits: 1,
        sinks: true,
        gate: false,
        segmented: false,
    };
    for case in [
        flint(1, 3199, CpfAttnSpan::FULL),
        flint(8, 3192, CpfAttnSpan::FULL),
        flint(
            8,
            4100,
            CpfAttnSpan {
                window: 513,
                ring: 2048,
            },
        ),
        // A sliding layer before the cache reaches the ring depth — the span.window planner
        // branch without the ring wrap.
        Case {
            shape: AttnPrefillShape {
                m: 8,
                n_head: 32,
                n_kv_head: 4,
                head_dim: 128,
                cache_rows: 512,
                ld: 0,
                p0: 504,
            },
            signed: true,
            group_size: 32,
            scalar: (Dtype::F16, None),
            span: CpfAttnSpan {
                window: 513,
                ring: 0,
            },
            splits: 1,
            sinks: true,
            gate: false,
            segmented: false,
        },
    ] {
        let kv_len = (case.shape.p0 + case.shape.m as i64) as usize;
        let plan = cpf_attn_launch_plan(
            CpfAttnGeometry {
                m: case.shape.m,
                n_head: case.shape.n_head,
                n_kv_head: case.shape.n_kv_head,
                head_dim: case.shape.head_dim,
            },
            kv_len,
            case.span.window as usize,
            CPF_ATTN_SM_COUNT,
        )
        .expect("launch plan");
        assert!(
            plan.max_splits > 1,
            "depth {kv_len}: no split raise to test"
        );
        let mut one = hoist_case(Case { splits: 1, ..case }, plan.min_keys, &[], false, false)
            .expect("splits1 run");
        let (out_one, ..) = one.pop().expect("reference path");
        let run_plan = || {
            let mut r = hoist_case(
                Case {
                    splits: plan.max_splits,
                    ..case
                },
                plan.min_keys,
                &[],
                false,
                false,
            )
            .expect("planner run");
            r.pop().expect("reference path").0
        };
        let out_plan = run_plan();
        assert_eq!(out_plan, run_plan(), "depth {kv_len}: not deterministic");
        let floats = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
        for (a, b) in floats(&out_one).iter().zip(floats(&out_plan)) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / a.abs().max(1.0));
        }
        assert!(max_abs <= 1e-3, "depth {kv_len}: max_abs {max_abs:e}");
        assert!(max_rel <= 1e-4, "depth {kv_len}: max_rel {max_rel:e}");
        eprintln!(
            "PLANNER-TOL depth {kv_len} splits {} max_abs {max_abs:.3e} max_rel {max_rel:.3e}",
            plan.max_splits
        );
    }
}
