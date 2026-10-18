//! Host-only tests of the MMQ sizing module (CPF-V4 G1): the exhaustive pad, ids and fixup
//! assertions over the whole chunk domain (rules M1-M3, rev 3), the strict cross-check that
//! re-parses both vendored config headers, the J-replica pins (the strict-`<` row, the table
//! gaps, the synthetic `smpbo` skip), the exact-equality pad contract, the Qwen width pins,
//! the recomputed under-pad windows of the old upstream rule, the ncols domain guard and the
//! typed overflow refusals. GPU-free but under `kernels::cuda`, which the crate compiles
//! only with `native-cuda` (kernels/mod.rs:26): the rows run with
//! `cargo test --features native-cuda`, never touching a device (the sister device tests
//! also need the feature, and a GPU besides).

use super::*;

/// One parsed CASE row: (type, nthreads, occupancy, I, J, layout, K_vram, stream_k,
/// fallback).
type CaseRow = (
    String,
    usize,
    usize,
    usize,
    usize,
    String,
    usize,
    bool,
    bool,
);

/// The strict `CASE(...)` field parser: any line of a config header that mentions CASE must
/// parse completely, or the test fails (finding 7: a vendor patch cannot pass silently).
fn parse_cases(text: &str) -> Vec<CaseRow> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if !line.contains("CASE(") {
            continue;
        }
        let fields: Vec<&str> = line
            .trim()
            .trim_start_matches("CASE(")
            .trim_end_matches(");")
            .split(',')
            .map(str::trim)
            .collect();
        assert!(
            fields.len() == 9,
            "unparsable CASE line (expected 9 fields): {line}"
        );
        let (type_, nthreads, occupancy, i_tile, j) = (
            fields[0].to_string(),
            fields[1]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("bad nthreads: {line}")),
            fields[2]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("bad occupancy: {line}")),
            fields[3]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("bad I: {line}")),
            fields[4]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("bad J: {line}")),
        );
        assert!(
            fields[6] == "MMQ_ITER_K",
            "unexpected K_vram field (the table moved off MMQ_ITER_K): {line}"
        );
        let layout = fields[5].to_string();
        let stream_k = fields[7]
            .parse::<bool>()
            .unwrap_or_else(|_| panic!("bad stream_k: {line}"));
        let fallback = fields[8]
            .parse::<bool>()
            .unwrap_or_else(|_| panic!("bad fallback: {line}"));
        rows.push((
            type_, nthreads, occupancy, i_tile, j, layout, 256, stream_k, fallback,
        ));
    }
    rows
}

/// The opt-in shared-memory budget of `cc` in the tests: 96 KiB on sm_72, 99 KiB on the
/// 3090 (the values of §B.0 (a); the skip never fires on either).
fn smpbo_of(cc: i32) -> usize {
    if cc >= 800 { 101_376 } else { 98_304 }
}

/// The layout string the vendored CASE rows name per type (both headers agree).
fn layout_of(mmq_type: MmqType) -> &'static str {
    match mmq_type {
        MmqType::Q80 | MmqType::Q50 => "GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0",
        MmqType::Q4K => "GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1",
        MmqType::Q6K => "GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K",
    }
}

fn type_name(mmq_type: MmqType) -> &'static str {
    match mmq_type {
        MmqType::Q80 => "GGML_TYPE_Q8_0",
        MmqType::Q50 => "GGML_TYPE_Q5_0",
        MmqType::Q4K => "GGML_TYPE_Q4_K",
        MmqType::Q6K => "GGML_TYPE_Q6_K",
    }
}

/// Every CASE row of both vendored config headers must equal this module's tables (finding
/// 7): the parsed (type, nthreads, I, J, fallback) sets, the layout, the occupancy, K_vram
/// and stream_k. A vendor patch that changes a row fails here without a device.
#[test]
fn config_tables_match_vendored_headers() {
    let headers = [
        (
            MmqConfig::Ampere,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-ampere.cuh"
            )),
        ),
        (
            MmqConfig::PascalDp4a,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-pascal-dp4a.cuh"
            )),
        ),
    ];
    for mmq_type in [MmqType::Q80, MmqType::Q50, MmqType::Q4K, MmqType::Q6K] {
        for (table, text) in headers {
            let cfg = table.config();
            for (fallback, j_set) in cfg.j_sets.iter().enumerate() {
                let fallback = fallback == 1;
                let mut parsed: Vec<usize> = parse_cases(text)
                    .into_iter()
                    .filter(|row| row.0 == type_name(mmq_type) && row.8 == fallback)
                    .map(|row| {
                        assert_eq!(row.1, cfg.nthreads, "nthreads of {mmq_type:?} {table:?}");
                        assert_eq!(
                            row.2,
                            usize::from(table == MmqConfig::PascalDp4a) + 1,
                            "occupancy"
                        );
                        assert_eq!(row.3, cfg.i_tile, "I of {mmq_type:?} {table:?}");
                        assert_eq!(row.5, layout_of(mmq_type), "sram layout");
                        assert_eq!(row.7, table.stream_k(mmq_type, row.4), "stream_k");
                        row.4
                    })
                    .collect();
                parsed.sort_unstable();
                assert_eq!(
                    parsed.as_slice(),
                    *j_set,
                    "J-set of {mmq_type:?} fb={fallback} {table:?}"
                );
            }
        }
    }
}

/// The `smpbo` skip never fires on either build arch for any of our types (§B.0 (a)): the
/// largest shared need is re-derived from the table data, not trusted. sm_72 offers 96 KiB
/// and the 3090 99 KiB opt-in per block.
#[test]
fn shared_bytes_fit_both_build_architectures() {
    for (cc, smpbo, largest) in [(720, 98_304, 56_448), (860, 101_376, 57_856)] {
        let mut max = 0;
        for mmq_type in [MmqType::Q80, MmqType::Q50, MmqType::Q4K, MmqType::Q6K] {
            for fallback in [false, true] {
                for j in (8..=128).step_by(8) {
                    if let Some(nbytes) =
                        MmqConfig::select(cc).nbytes_shared(mmq_type, fallback, cc, j)
                    {
                        assert!(nbytes <= smpbo, "{mmq_type:?} J={j} cc={cc}: {nbytes}");
                        max = max.max(nbytes);
                    }
                }
            }
        }
        assert_eq!(max, largest, "largest shared need at cc={cc}");
    }
}

/// The replica's pinned rows: the strict-`<` tie at 130 (96/112/128 tie with 80 and lose)
/// and the table gaps of both fallback axes.
#[test]
fn j_best_pins_the_launch_selection() {
    for cc in [720, 860] {
        assert_eq!(j_best(MmqType::Q4K, false, cc, smpbo_of(cc), 130), Some(80));
        for (n, want) in [(56, 64), (72, 80), (88, 96), (104, 112), (120, 128)] {
            assert_eq!(
                j_best(MmqType::Q80, false, cc, smpbo_of(cc), n),
                Some(want),
                "n={n}"
            );
        }
        for (n, want) in [(24, 32), (40, 64), (120, 128)] {
            assert_eq!(
                j_best(MmqType::Q6K, true, cc, smpbo_of(cc), n),
                Some(want),
                "n={n}"
            );
        }
        // Membership: every answer is a table J, for the whole chunk domain.
        for fallback in [false, true] {
            for n in 1..=NCOLS_LIMIT {
                let j = j_best(MmqType::Q50, fallback, cc, smpbo_of(cc), n).unwrap();
                assert!(MmqConfig::select(cc).contains(fallback, j), "n={n} j={j}");
            }
        }
    }
}

/// The synthetic `smpbo` row (§B.0 (a)): a tiny opt-in forces the skip and shrinks J, and a
/// value no J fits is the launch's refusal (`None`), never an out-of-bounds guess.
#[test]
fn j_best_models_the_shared_memory_skip() {
    // cc=860, Q4_K, fb=false: shared(96) = 53,632, shared(112) = 55,744, shared(128) = 57,856;
    // the ampere table has no J = 104, so with 112 skipped the best table J is 96.
    assert_eq!(j_best(MmqType::Q4K, false, 860, 56_000, 2048), Some(112));
    assert_eq!(j_best(MmqType::Q4K, false, 860, 55_000, 2048), Some(96));
    assert_eq!(j_best(MmqType::Q4K, false, 860, 101_376, 2048), Some(128));
    assert_eq!(j_best(MmqType::Q4K, false, 860, 1_000, 2048), None);
}

/// The exact-equality pad contract (rev 3, finding 19): `Q8_1_PAD == 18,432` and
/// `q8_1_bytes == (Kpad/128)*ncols*144 + 18,432`, at dense, grouped and non-multiple K
/// shapes. V18 (floored Kpad) dies at K = 600; V2/V3 die at the dense (2048, 1) row, where
/// their `ncols_opt`-sized pad is 2,048 B against the contract's 18,432 B.
#[test]
fn q8_1_pad_contract_is_exact() {
    assert_eq!(Q8_1_PAD, 18_432);
    assert_eq!(Q8_1_PAD, 9 * SRC1_LOAD_CHUNK);
    let cases = [
        (2048_usize, 1_usize),
        (512, 1),
        (4096, 1),
        (600, 1), // Kpad rounds up to 640: 5 blocks, not 4
        (129, 1), // two blocks
        (2048, 2048),
        (2048, 8 * 49),  // grouped, M = 49: the B1 row
        (2048, 8 * 56),  // grouped, M = 56
        (512, 8 * 2048), // the q8_act row at the cap
    ];
    for (k, ncols) in cases {
        let blocks = k.div_ceil(128);
        assert_eq!(
            q8_1_bytes(k, ncols).unwrap(),
            blocks * ncols * 144 + 18_432,
            "K={k} ncols={ncols}"
        );
    }
    for (k, ncols) in [(0, 1), (1, 0)] {
        let err = q8_1_bytes(k, ncols).unwrap_err();
        assert!(matches!(err, EngError::BadRequest(ref m) if m.contains("mmq_size: zero extent")));
    }
}

/// The Qwen width pins (§B.3 rule M1, per token before the pad) and the ids rows of §C.3.
#[test]
fn qwen_widths_and_ids_pins() {
    assert_eq!(q8_1_bytes(2048, 1).unwrap(), 2_304 + 18_432); // q8_x, q8_x2, q8_s
    assert_eq!(q8_1_bytes(512, 1).unwrap(), 576 + 18_432); // q8_ash, q8_act
    assert_eq!(q8_1_bytes(4096, 1).unwrap(), 4_608 + 18_432); // q8_o_d, q8_o_a
    assert_eq!(q8_1_bytes(2048, 8).unwrap(), 18_432 + 18_432); // scatter per token
    for m in [128, 512, 2048] {
        assert_eq!(ids_dst_ints(8 * m, m).unwrap(), 9 * m + 127);
        assert_eq!(ids_src1_ints(8 * m), 8 * m);
    }
    assert_eq!(expert_bounds_ints(256).unwrap(), 257);
    assert_eq!(fixup_floats_max(82).unwrap(), 82 * 128 * 128);
    assert_eq!(fixup_floats_max(82).unwrap() * 4, 5_373_952); // 5.125 MiB
}

/// The exhaustive dense assertion: `S(J_best(n)) - r*144 <= Q8_1_PAD` for every n in
/// 1..=2048, both fallbacks, both build archs, with the minimum margin witnessed.
#[test]
fn exhaustive_dense_pad_rule() {
    for cc in [720, 860] {
        let nthreads = MmqConfig::select(cc).config().nthreads;
        for fallback in [false, true] {
            let mut worst = 0;
            for n in 1..=NCOLS_LIMIT {
                let j = j_best(MmqType::Q4K, fallback, cc, smpbo_of(cc), n).unwrap();
                let need = y_tile_span_bytes(nthreads, j) - tail_r(n, j) * 144;
                assert!(need <= Q8_1_PAD, "n={n} fb={fallback} j={j} need={need}");
                if j == 128 {
                    // margin = Q8_1_PAD - need = r*144, minimum 144 at r = 1.
                    assert_eq!(Q8_1_PAD - need, tail_r(n, j) * 144);
                }
                worst = worst.max(need);
            }
            assert_eq!(worst, 18_288, "worst need fb={fallback} cc={cc}");
        }
    }
}

/// The exhaustive grouped assertion: the last tile can hold a single-column top expert, so
/// `S(J_best(M)) - 144 <= Q8_1_PAD` for every M in 1..=2048. The B1 witness: M = 49..56 at
/// fallback = false picks J = 64, needs 9,072 B, and rev 2's pad covered only 8,192 B.
#[test]
fn exhaustive_grouped_pad_rule() {
    for cc in [720, 860] {
        let nthreads = MmqConfig::select(cc).config().nthreads;
        for fallback in [false, true] {
            for m in 1..=NCOLS_LIMIT {
                let j = j_best(MmqType::Q4K, fallback, cc, smpbo_of(cc), m).unwrap();
                let need = y_tile_span_bytes(nthreads, j) - 144;
                assert!(need <= Q8_1_PAD, "M={m} fb={fallback} j={j} need={need}");
            }
        }
        for m in 49..=56 {
            let j = j_best(MmqType::Q4K, false, cc, smpbo_of(cc), m).unwrap();
            assert_eq!(j, 64, "M={m}");
            assert_eq!(y_tile_span_bytes(256, 64) - 144, 9_072);
            let rev2 = pad_up(j_ub(m) * 144, SRC1_LOAD_CHUNK);
            assert_eq!(rev2, 8_192, "rev 2's pad at M={m}");
            assert_eq!(9_072 - rev2, 880, "the B1 shortfall at M={m}");
        }
    }
}

/// The typed underflow text of `ids_gather_ints(0, 0)` (review NIT-5): its own rule, not the
/// overflow text.
#[test]
fn ids_gather_zero_extents_have_their_own_text() {
    let err = ids_gather_ints(0, 0).unwrap_err();
    assert!(
        matches!(err, EngError::BadRequest(ref m) if m.contains("mmq_size: zero extent")
            && m.contains("ids_gather_ints(0, 0) underflows")),
        "{err:?}"
    );
    assert!(
        matches!(ids_gather_ints(usize::MAX, 1).unwrap_err(), EngError::BadRequest(ref m)
            if m.contains("mmq_size: size overflow"))
    );
}

/// The exhaustive ids and fixup forms over the reachable domain (M up to 2,048,
/// `ne_get_rows` up to 16,384): the shipped closed form bounds both gathers at the J the
/// launch picks, for every M, fallback and arch. The V1 regression: at `ncols_opt = 24`
/// with fallback, `J_ub = 24` but `J_best = 32`, so rev 2's `ne_get_rows + J_ub - 1` form
/// is 8 ints short of the fixup gather's exact reach.
#[test]
fn exhaustive_ids_and_fixup_forms() {
    for cc in [720, 860] {
        for fallback in [false, true] {
            for m in 1..=NCOLS_LIMIT {
                let ne_get_rows = 8 * m;
                let j = j_best(MmqType::Q4K, fallback, cc, smpbo_of(cc), m).unwrap();
                let closed = ids_dst_ints(ne_get_rows, m).unwrap();
                assert_eq!(closed, 9 * m + 127);
                assert!(
                    closed >= ids_gather_ints(ne_get_rows + m, j).unwrap(),
                    "M={m}"
                );
                assert!(closed >= ids_gather_ints(ne_get_rows, j).unwrap(), "M={m}");
            }
        }
    }
    assert_eq!(j_best(MmqType::Q4K, true, 860, 101_376, 24), Some(32));
    assert_eq!(j_ub(24), 24);
    assert_eq!(legacy_j_max(true, 860, 24), 16);
    let short = ids_gather_ints(8 * 24 + 24, 24).unwrap();
    let exact = ids_gather_ints(8 * 24 + 24, 32).unwrap();
    assert_eq!(
        exact - short,
        8,
        "rev 2's form is 8 ints short at ncols_opt=24"
    );
    assert!(fixup_floats_max(82).unwrap() >= fixup_floats_exact(82, 128, 128).unwrap());
    assert!(fixup_floats_max(82).unwrap() > fixup_floats_exact(61, 80, 128).unwrap());
}

/// The recomputed trigger windows (finding 9): the old helper under-pads exactly
/// `{1..7, 9..13}` at fallback = false and adds `{17, 18, 19}` at fallback = true, over the
/// whole domain, on both build archs. The old rule is wrong exactly on these rows: 14 and
/// 20 stay safe, and the fb=true gap identity of finding 3 holds (`J_max = 16` against
/// `J_best = 32` at 24).
#[test]
fn overread_windows_recomputed() {
    let plain: Vec<usize> = (1..=7).chain(9..=13).collect();
    let mut with_gaps = plain.clone();
    with_gaps.extend([17, 18, 19]);
    for cc in [720, 860] {
        for mmq_type in [MmqType::Q80, MmqType::Q50, MmqType::Q4K, MmqType::Q6K] {
            assert_eq!(
                overread_window(mmq_type, false, cc, smpbo_of(cc)),
                plain,
                "fb=false cc={cc}"
            );
            assert_eq!(
                overread_window(mmq_type, true, cc, smpbo_of(cc)),
                with_gaps,
                "fb=true cc={cc}"
            );
        }
    }
    assert!(legacy_j_max(false, 860, 14) >= 8); // 14 is safe: the old pad covers the need
}

/// The domain guard (finding 22): 1..=2048 passes, 0 and 2049 are typed refusals that a
/// wrapper refusal test can pin.
#[test]
fn ncols_domain_guard() {
    assert!(check_ncols_domain(1).is_ok());
    assert!(check_ncols_domain(2048).is_ok());
    for ncols in [0, 2049] {
        let err = check_ncols_domain(ncols).unwrap_err();
        assert!(
            matches!(err, EngError::BadRequest(ref m) if m.contains("mmq_size: chunk cap")),
            "ncols={ncols}"
        );
    }
}

/// Every sizing overflow is a typed `BadRequest` (checked arithmetic throughout).
#[test]
fn overflow_refusals_are_typed() {
    let near = usize::MAX;
    for result in [
        q8_1_bytes(near, 2),
        q8_1_bytes(2048, near),
        ids_dst_ints(near, 2),
        ids_gather_ints(near, 5),
        expert_bounds_ints(near),
        fixup_floats_max(near),
        fixup_floats_exact(near, 2, 2),
    ] {
        assert!(
            matches!(result, Err(EngError::BadRequest(ref m)) if m.contains("mmq_size: size overflow")),
            "{result:?}"
        );
    }
}

/// The table function of a config header must be EXACTLY the known shape (lane C's
/// completeness rule): no preprocessor line and no block comment may appear, and every
/// non-blank line is the one signature line, a CASE row, the known separator comment, or
/// the two-line footer. A vendor edit that hides a row behind `#ifdef` or a comment fails
/// here even when every CASE line still parses.
#[test]
fn config_headers_have_no_hidden_lines() {
    let headers = [
        (
            "ampere",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-ampere.cuh"
            )),
        ),
        (
            "pascal-dp4a",
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq-config-pascal-dp4a.cuh"
            )),
        ),
    ];
    const SEPARATOR: &str = "// ---------------------------------------------------------------------------------------------";
    let signature = |name: &str| {
        let suffix = if name == "ampere" {
            "ampere"
        } else {
            "pascal_dp4a"
        };
        format!(
            "static constexpr __host__ __device__ ggml_cuda_mmq_config \
ggml_cuda_mmq_get_config_{suffix}(ggml_type type, int J, bool fallback) {{"
        )
    };
    for (name, text) in headers {
        let mut saw_signature = false;
        let mut saw_footer = false;
        for line in text.lines() {
            let trimmed = line.trim();
            assert!(!line.contains('#'), "{name}: a preprocessor line: {line}");
            assert!(
                !trimmed.starts_with("/*") && !trimmed.starts_with("*"),
                "{name}: a block comment: {line}"
            );
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("CASE(") {
                assert!(!saw_footer, "{name}: a CASE row after the footer: {line}");
                continue; // parse_cases covers the fields
            }
            if !saw_signature {
                assert_eq!(
                    trimmed,
                    signature(name),
                    "{name}: the header's first line changed"
                );
                saw_signature = true;
                continue;
            }
            if trimmed == SEPARATOR {
                continue;
            }
            if trimmed.starts_with("return ggml_cuda_mmq_config(GGML_TYPE_COUNT,") {
                assert!(saw_signature, "{name}: footer before the signature");
                saw_footer = true;
                continue;
            }
            assert_eq!(trimmed, "}", "{name}: an unexpected line: {line}");
            assert!(saw_footer, "{name}: a stray close brace");
        }
        assert!(
            saw_signature && saw_footer,
            "{name}: missing head or footer"
        );
    }
}

/// The vendored J-selection loop (mmq.cuh:1444-1461) hash-pinned: the replica `j_best`
/// models this exact loop, so a vendor change to it must fail CI. The first line is
/// asserted first, so a line-number drift fails by text before the hash can mismatch.
#[test]
fn vendored_j_selection_loop_is_pinned() {
    let mmq = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq.cuh"
    ));
    let lines: Vec<&str> = mmq.lines().collect();
    let first = lines[1443];
    assert!(
        first.trim_start().starts_with("for (int J = 8; J <= 128"),
        "mmq.cuh:1444 is no longer the J loop: {first}"
    );
    let text: String = lines[1443..=1460]
        .iter()
        .fold(String::new(), |acc, line| acc + line + "\n");
    use sha2::Digest;
    let hash = sha2::Sha256::digest(text.as_bytes());
    assert_eq!(
        format!("{hash:x}"),
        "2eb7e09b7446e16f9d210888d2407d16d075817bcd73148fb1d4b73f7c8fc202",
        "the vendored J-selection loop changed; re-derive j_best against it"
    );
}

/// The sha-256 of vendored mmq.cuh lines `first..=last` (1-based), each with its newline,
/// after asserting that line `first` still starts with `head`: a line-number drift fails by
/// text before the hash can mismatch.
fn vendored_mmq_range_hash(first: usize, last: usize, head: &str) -> String {
    let mmq = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/vendor/ggml-cuda-mmq/src/ggml-cuda/mmq.cuh"
    ));
    let lines: Vec<&str> = mmq.lines().collect();
    assert!(
        lines[first - 1].trim_start().starts_with(head),
        "mmq.cuh:{first} no longer starts with {head:?}: {}",
        lines[first - 1]
    );
    let text: String = lines[first - 1..last]
        .iter()
        .fold(String::new(), |acc, line| acc + line + "\n");
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(text.as_bytes()))
}

/// The vendored stream-k launch rule (mmq.cuh:1395-1412): the 90 % tiling test, the `2^30`
/// limit, the fixup test and the fixup byte need. G2's `fixup_need` (C) and the Rust
/// mirrors of the 2^30 limit re-express exactly this passage, so a vendor change to it
/// must fail CI (lane C's ruling on the G2 verdict, m-1).
#[test]
fn vendored_stream_k_launch_rule_is_pinned() {
    assert_eq!(
        vendored_mmq_range_hash(1395, 1412, "// For the stream-k kernel it is possible"),
        "38ebc22e187c1deecd6a4afb30c0171e8dfecf988cadbf9da22ee59f07477db8",
        "the vendored stream-k rule changed; re-derive fixup_need and the grid-limit mirror"
    );
}

/// The vendored launch's grid setup and stream-k branch test (mmq.cuh:1365-1367 and :1385):
/// the row and column tile counts and the channel-sample count, and the test that picks the
/// stream-k kernel. `fixup_need` re-expresses both (tiles = ntx * nty with ntzw = 1; the
/// replica returns 0 for a type or J without a stream-k row), so a vendor change must fail CI.
#[test]
fn vendored_launch_grid_and_stream_k_branch_are_pinned() {
    assert_eq!(
        vendored_mmq_range_hash(1365, 1367, "const int nty"),
        "92188104c7dbffb6eb6d136a8367c88af3a1901f2e4849412f61a87324cc1a60",
        "the vendored tile-count setup changed; re-derive fixup_need"
    );
    assert_eq!(
        vendored_mmq_range_hash(
            1385,
            1385,
            "if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc))"
        ),
        "880850ab75e67f60c46b55e0b0ef3374612189071d731723f9b24a00dea9ed0d",
        "the vendored stream-k branch test changed; re-derive fixup_need"
    );
}

/// The vendored case function (mmq.cuh:1500-1510): the device check and the fallback rule,
/// `nrows_x % 128 != 0`, that picks the J set. Every sizing row names the fallback axis from
/// it (plan section B.3), so a vendor change to it must fail CI.
#[test]
fn vendored_mmq_case_fallback_rule_is_pinned() {
    assert_eq!(
        vendored_mmq_range_hash(1500, 1510, "template <ggml_type type>"),
        "3160816f078b9a79c9ff6a131ac94bbb0b407e1acdd3f12c734bd10272481b79",
        "the vendored mul_mat_q_case changed; re-derive the fallback axis of the sizing"
    );
}

/// `COMPILED_ARCHS` is pinned to the build (lane C's item 4): every architecture THIS build
/// compiled — the list build.rs wrote into its generated `__CUDA_ARCH_LIST__` header — must
/// appear in the constant, so a build whose arch set grows past it fails here instead of
/// silently taking the wrong config table. (Deriving the constant from
/// `INFERENCE_CUDA_ARCHS` at run time is not possible: the test exes run outside the build
/// environment, where the variable is absent.)
#[test]
fn compiled_archs_cover_the_build() {
    // Embedded at build time, not read at run time: a test exe copied away from its target
    // directory (a mutant's, a sanitizer's) still has the list.
    let generated = include_str!(concat!(env!("OUT_DIR"), "/fifi_cuda_arch_list.h"));
    let list = generated
        .lines()
        .find_map(|line| line.strip_prefix("#define __CUDA_ARCH_LIST__ "))
        .expect("the __CUDA_ARCH_LIST__ define");
    let built: Vec<i32> = list
        .split(',')
        .map(|arch| {
            arch.trim()
                .parse::<i32>()
                .unwrap_or_else(|error| panic!("arch {arch:?}: {error}"))
        })
        .collect();
    assert!(!built.is_empty(), "no compiled architecture");
    for cc in built {
        assert!(
            COMPILED_ARCHS.contains(&cc),
            "the build compiled cc {cc}, outside COMPILED_ARCHS {:?}",
            COMPILED_ARCHS
        );
    }
}
