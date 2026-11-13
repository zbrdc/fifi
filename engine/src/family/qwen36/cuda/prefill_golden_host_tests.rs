//! The host half of the golden harness: digests of the plan tables the prefill promotion moves —
//! the EXL3 chunk tensor list and arena offsets, the family (GGUF) table beside it, the planned
//! table the arena build must realize, the zero-fill spans at the served granularity, the dense
//! job tables, the admission rung over a budget grid built from the admission's own thresholds,
//! and the golden prompt's partition — all at the real Qwen3.6 35B-A3B dimensions and the RTX
//! 3090's 82 SMs. Pure host computation, but still compiled `--features native-cuda` in the
//! container: name this test in every promotion battery.
//!
//! Every promotion commit that moves or parameterises any of this must keep these digests, and
//! every device arm of `prefill_golden_device_tests` must keep matching the planned table and the
//! partition pinned here.
//!
//! `FIFI_GOLDEN_UPDATE=1` rewrites the `host` section of `testdata/prefill-goldens.json`
//! atomically instead of comparing (used once at capture; the file is committed). The update
//! reads the committed file from disk, not the compile-time pin, so device arms the capture
//! campaign wrote earlier survive the host update.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::common_tests::{dims, plan_of};
use super::dims::KvMode;
use super::prefill::exl3::dense_jobs;
use super::prefill_driver::{AdmitFacts, ZERO_CHUNK, admit_m_sess, zero_spans};
use super::prefill_golden_device_tests::{
    FED_ROWS, IDS_SHA256, PROMPT_IDS, planned_table, prompt_ids, store_goldens, table_text,
};
use super::prefill_plan::{
    M_LADDER, PART_CHARGE, PrefillStep, SPARE_MIN, chunk_list_with_depth_tier, chunk_tensors,
    exl3_chunk_costs, exl3_chunk_tensors, state_bytes,
};
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::plan_arena;

/// The SM count the family digests are attributed to: the RTX 3090 of the device arms. The
/// runtime reads it from the device (`MmqFacts::nsm`); a digest names its input.
const RTX3090_NSM: usize = 82;

/// The chunk sizes every table is digested at: the ladder, the special 16, and 256.
const SIZES: [usize; 6] = [16, 128, 256, 512, 1024, 2048];

/// The caps whose partition the golden prompt pins a one-row `Feed` tail for.
const FEED_TAIL_CAPS: [usize; 3] = [16, 128, 256];

/// The caps whose partition ends in the 257-row partial chunk.
const PARTIAL_TAIL_CAPS: [usize; 3] = [512, 1024, 2048];

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Appends the section line to `body` and answers the section's digest.
fn section(body: &mut String, name: &str, rendered: &str) -> String {
    let sha = digest(rendered);
    let _ = writeln!(body, "{name} sha256 {sha}");
    let _ = writeln!(body, "{rendered}");
    sha
}

#[test]
fn the_plan_tables_hold_their_golden_digests() {
    let dims = dims();
    assert_eq!(prompt_ids().len(), PROMPT_IDS, "the committed ids fixture");
    let mut body = String::new();
    let mut sections = serde_json::Map::new();

    // The EXL3 chunk tensors per size with their arena offsets, and the family (GGUF) table
    // beside it: numeric lines, one per tensor.
    for (name, exl3) in [("exl3_tables", true), ("family_tables", false)] {
        let mut text = String::new();
        for &m in &SIZES {
            let list = if exl3 {
                exl3_chunk_tensors(&dims, m).expect("exl3 list")
            } else {
                chunk_tensors(&dims, m, RTX3090_NSM).expect("family list")
            };
            let arena =
                plan_arena(&list.iter().map(|&(_, t)| t).collect::<Vec<_>>()).expect("arena");
            let _ = write!(text, "{}", table_text(&list, arena.total, m));
            let offsets: Vec<String> = arena.offsets.iter().map(|o| o.to_string()).collect();
            let _ = writeln!(text, "offsets {}", offsets.join(","));
        }
        let sha = section(&mut body, name, &text);
        sections.insert(name.into(), sha.into());
    }

    // The planned table the arena build must realize (the device arms compare the table their
    // session really built against this text): one digest per size, both layouts.
    for (name, layout) in [
        ("arena_built_exl3", WeightLayout::Exl3),
        ("arena_built_family", WeightLayout::Native),
    ] {
        let mut text = String::new();
        for &m in &SIZES {
            let _ = write!(text, "{}", planned_table(&dims, RTX3090_NSM, m, layout));
        }
        let sha = section(&mut body, name, &text);
        sections.insert(name.into(), sha.into());
    }

    // The zero-fill spans of every size's arena at the served granularity: the real call is
    // `zero_spans(total, ZERO_CHUNK)`. Compact numeric lines — count, first, last, and a running
    // hash over the span tuples — never the whole span list.
    let mut zero_text = String::new();
    for &m in &SIZES {
        let list = exl3_chunk_tensors(&dims, m).expect("exl3 list");
        let total = plan_arena(&list.iter().map(|&(_, t)| t).collect::<Vec<_>>())
            .expect("arena")
            .total;
        let spans = zero_spans(total, ZERO_CHUNK).expect("zero spans");
        let mut running = Sha256::new();
        for (at, len) in &spans {
            running.update(format!("{at}:{len}\n").as_bytes());
        }
        let first = spans.first().expect("at least one span");
        let last = spans.last().expect("at least one span");
        let _ = writeln!(
            zero_text,
            "m={m} total={total} count={} first={}:{} last={}:{} span_sha256={:x}",
            spans.len(),
            first.0,
            first.1,
            last.0,
            last.1,
            running.finalize()
        );
        // The spans must cover [0, total) exactly once, contiguous, each at most ZERO_CHUNK.
        let mut at = 0;
        for (span_at, span_len) in &spans {
            assert_eq!(*span_at, at, "contiguous coverage from 0");
            assert!(*span_len <= ZERO_CHUNK, "the served granularity");
            at += span_len;
        }
        assert_eq!(at, total, "the whole arena is zeroed");
    }
    let sha = section(&mut body, "zero_spans", &zero_text);
    sections.insert("zero_spans".into(), sha.into());

    // The dense job tables: rows cut into tiles of the job height.
    let mut jobs_text = String::new();
    for &rows in &SIZES {
        let jobs = dense_jobs(rows);
        let first = jobs.first().expect("at least one job");
        let last = jobs.last().expect("at least one job");
        let _ = writeln!(
            jobs_text,
            "rows={rows} jobs={} first_member={} first_row0={} first_rows={} last_row0={} last_rows={}",
            jobs.len(),
            first.member,
            first.row0,
            first.rows,
            last.row0,
            last.rows
        );
        assert_eq!(
            jobs.iter().map(|job| job.rows as usize).sum::<usize>(),
            rows,
            "the jobs cover the rows exactly once"
        );
    }
    let sha = section(&mut body, "dense_jobs", &jobs_text);
    sections.insert("dense_jobs".into(), sha.into());

    // The admission rung over budgets built from the admission's own thresholds: for each ladder
    // size, the budget that admits it at exactly the spare minimum, one byte less, and enough for
    // the whole ladder at three caps. The decode arena is the EXL3 one-row plan the device
    // charges (not the Tiled one); the K-row arena term is zero in this grid, a named input.
    let costs = exl3_chunk_costs(&dims, RTX3090_NSM).expect("costs");
    let decode_arena = plan_of(&dims, WeightLayout::Exl3).arena_bytes();
    let state = state_bytes(&dims, KvMode::Fp16, 8_192, 1).expect("state bytes");
    let thresholds: Vec<(usize, usize)> = costs
        .iter()
        .map(|cost| {
            (
                cost.m,
                state + decode_arena + cost.arena + PART_CHARGE.max(cost.part) + SPARE_MIN,
            )
        })
        .collect();
    let mut admit_text = String::new();
    let _ = writeln!(
        admit_text,
        "state={state} decode_arena={decode_arena} part_charge={PART_CHARGE} spare_min={SPARE_MIN}"
    );
    for &(m, threshold) in &thresholds {
        let _ = writeln!(admit_text, "m={m} threshold={threshold}");
    }
    // The expected rung under the threshold table: the largest ladder size at most the cap whose
    // threshold fits the budget, `None` when none does.
    let expected = |budget: usize, cap: usize| -> Option<usize> {
        thresholds
            .iter()
            .filter(|&&(m, threshold)| m <= cap && threshold <= budget)
            .map(|&(m, _)| m)
            .max()
    };
    let mut rungs_seen: Vec<Option<usize>> = Vec::new();
    let mut budgets: Vec<usize> = vec![0];
    budgets.extend(thresholds.iter().map(|&(_, t)| t - 1));
    budgets.extend(thresholds.iter().map(|&(_, t)| t));
    let room_for_all = thresholds.iter().map(|&(_, t)| t).max().unwrap() + (1 << 30);
    for cap in [2048, 512, 128] {
        for &budget in &budgets {
            let facts = AdmitFacts {
                budget,
                decode_arena,
                state_planned: state,
                state_allocated: state,
                cap,
            };
            let rung = admit_m_sess(&facts, &[], &costs).expect("admit");
            assert_eq!(rung, expected(budget, cap), "budget={budget} cap={cap}");
            if cap == 2048 {
                rungs_seen.push(rung);
            }
            let _ = writeln!(admit_text, "budget={budget} cap={cap} rung={rung:?}");
        }
        let facts = AdmitFacts {
            budget: room_for_all,
            decode_arena,
            state_planned: state,
            state_allocated: state,
            cap,
        };
        let rung = admit_m_sess(&facts, &[], &costs).expect("admit");
        assert_eq!(rung, Some(cap), "a cap-bound roomy budget admits the cap");
        let _ = writeln!(admit_text, "budget=roomy cap={cap} rung={rung:?}");
    }
    for rung in M_LADDER {
        assert!(
            rungs_seen.contains(&Some(rung)),
            "the grid must admit every ladder rung ({rung})"
        );
    }
    assert!(
        rungs_seen.contains(&None),
        "the grid must include a budget that admits nothing"
    );
    let sha = section(&mut body, "admission_grid", &admit_text);
    sections.insert("admission_grid".into(), sha.into());

    // The golden prompt's partition, per cap: the stage rows the device arms assert. `FED_ROWS`
    // is fed; the token after them goes through decode.
    let mut partition_text = String::new();
    for &cap in &SIZES {
        let steps = chunk_list_with_depth_tier(0, FED_ROWS, cap, false).expect("chunk list");
        for step in &steps {
            match step {
                PrefillStep::Chunk { p0, rows } => {
                    let _ = writeln!(partition_text, "cap={cap} chunk p0={p0} rows={rows}");
                }
                PrefillStep::Feed { p0 } => {
                    let _ = writeln!(partition_text, "cap={cap} feed p0={p0} rows=1");
                }
            }
        }
        assert_eq!(
            steps.iter().map(step_rows).sum::<usize>(),
            FED_ROWS,
            "cap={cap}: the steps cover the fed rows"
        );
        if FEED_TAIL_CAPS.contains(&cap) {
            assert!(
                matches!(steps.last(), Some(PrefillStep::Feed { .. })),
                "cap={cap}: the tail must be a one-row feed"
            );
        }
        if PARTIAL_TAIL_CAPS.contains(&cap) {
            assert!(
                matches!(steps.last(), Some(PrefillStep::Chunk { rows: 257, .. })),
                "cap={cap}: the tail must be the 257-row partial chunk"
            );
        }
    }
    // At cap 2048 the depth tier, not the cap, cuts the chunks from position 4,096 on.
    let deep = chunk_list_with_depth_tier(0, FED_ROWS, 2048, false).expect("chunk list");
    assert!(
        deep.iter().any(|step| matches!(
            step,
            PrefillStep::Chunk {
                p0: 4096,
                rows: 1024
            }
        )),
        "cap=2048: the chunk at 4,096 must be cut to 1,024 rows by the depth tier"
    );
    let sha = section(&mut body, "partition", &partition_text);
    sections.insert("partition".into(), sha.into());

    // The committed goldens, pinned into the binary at compile time: a drifted checkout cannot
    // pass a compare it should fail.
    const GOLDENS_TEXT: &str = include_str!("testdata/prefill-goldens.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/family/qwen36/cuda/testdata/prefill-goldens.json");
    if std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty()) {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let mut goldens: serde_json::Value =
            serde_json::from_str(&text).expect("the committed goldens on disk");
        goldens["host"] = serde_json::Value::Object(sections);
        goldens["host"]["whole"] = digest(&body).into();
        goldens["host"]["prompt_sha256"] = IDS_SHA256.into();
        goldens["host"]["fed_rows"] = FED_ROWS.into();
        store_goldens(&path, goldens);
        println!("UPDATED host goldens in {}", path.display());
    } else {
        let pinned: serde_json::Value = serde_json::from_str(GOLDENS_TEXT)
            .expect("the committed host goldens; capture with FIFI_GOLDEN_UPDATE=1");
        assert!(
            !pinned["host"].is_null(),
            "no host goldens; capture with FIFI_GOLDEN_UPDATE=1"
        );
        let mut drifted: Vec<String> = Vec::new();
        if pinned["host"]["whole"].as_str() != Some(digest(&body).as_str()) {
            for (name, sha) in &sections {
                let computed = sha.as_str().unwrap_or("");
                let pinned_sha = pinned["host"][name].as_str().unwrap_or("");
                if pinned_sha != computed {
                    drifted.push(format!(
                        "{name}: pinned {pinned_sha:?}, computed {computed}"
                    ));
                }
            }
            if drifted.is_empty() {
                drifted.push("whole digest only: a section was added or removed".into());
            }
        }
        assert!(
            drifted.is_empty(),
            "plan tables the promotion must keep have moved; sections:\n{}",
            drifted.join("\n")
        );
        assert_eq!(
            pinned["host"]["prompt_sha256"].as_str(),
            Some(IDS_SHA256),
            "the host goldens were captured against a different prompt fixture"
        );
        println!("GOLDEN MATCH host");
    }
}

/// The rows a partition step feeds.
fn step_rows(step: &PrefillStep) -> usize {
    match step {
        PrefillStep::Chunk { rows, .. } => *rows,
        PrefillStep::Feed { .. } => 1,
    }
}
