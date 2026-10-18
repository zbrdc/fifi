//! T0b: cold read rate of single experts from the checkpoint (S3 plan section 3.6; Phase 0B
//! deliverable 4 for experts).
//!
//! ```text
//! expert-reads --artifact <checkpoint dir> --out <dir> [--experts 64] [--depths 1,2,4,8,16]
//!     [--seed 20261009] [--min-cold 0.9] [--prefix language_model.model.]
//! ```
//!
//! The command needs no privileges and never purges a cache. It draws experts in a seeded order
//! and reads only those whose bytes `mincore` reports at least `--min-cold` not resident, checked
//! right before the read. Each case reads `--experts` experts in groups of `depth` parallel
//! threads, one expert per thread, each expert as nine positioned reads (`pread`). The cases are,
//! first and on macOS only, cold reads through descriptors with `F_NOCACHE` at every depth, all on
//! one fixed set of experts (such reads leave the file cache unchanged, so the set stays cold);
//! then, for each depth, cold buffered reads of a set of their own and a warm re-read of the same
//! experts. A buffered read must leave the expert resident; otherwise `mincore` does not observe
//! the file cache and the result is invalid.

#![deny(unsafe_code)]

mod layout;
#[allow(unsafe_code)]
mod sys;

use inference_engine::weights::ShardHeader;
use layout::ExpertLayout;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Barrier;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use sys::Residency;

/// Largest safetensors header the command reads.
const HEADER_LIMIT: u64 = 64 << 20;

struct Settings {
    artifact: PathBuf,
    out: PathBuf,
    experts: usize,
    depths: Vec<usize>,
    seed: u64,
    min_cold: f64,
    prefix: String,
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1).collect()).and_then(|s| run(&s)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("expert-reads: {message}");
            ExitCode::FAILURE
        }
    }
}

fn parse(args: Vec<String>) -> Result<Settings, String> {
    let mut settings = Settings {
        artifact: PathBuf::new(),
        out: PathBuf::new(),
        experts: 64,
        depths: vec![1, 2, 4, 8, 16],
        seed: 20_261_009,
        min_cold: 0.9,
        prefix: "language_model.model.".to_string(),
    };
    let (mut artifact, mut out) = (None, None);
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        let value = iter.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--experts" => settings.experts = value.parse().map_err(|_| "--experts")?,
            "--depths" => {
                settings.depths = value
                    .split(',')
                    .map(|d| d.parse().map_err(|_| format!("depth {d}")))
                    .collect::<Result<_, _>>()?;
            }
            "--seed" => settings.seed = value.parse().map_err(|_| "--seed")?,
            "--min-cold" => settings.min_cold = value.parse().map_err(|_| "--min-cold")?,
            "--prefix" => settings.prefix = value,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    settings.artifact = artifact.ok_or("--artifact is required")?;
    settings.out = out.ok_or("--out is required")?;
    if settings.experts == 0 || settings.depths.contains(&0) || settings.depths.is_empty() {
        return Err("--experts and every depth must exceed 0".into());
    }
    Ok(settings)
}

/// Resident bytes of `offset..offset + len` from a `mincore` vector over pages that start at
/// `first`: the overlap of each resident page with the range.
pub fn resident_bytes(vec: &[u8], page: u64, first: u64, offset: u64, len: u64) -> u64 {
    let end = offset + len;
    vec.iter()
        .enumerate()
        .filter(|&(_, &status)| status & 1 == 1)
        .map(|(i, _)| {
            let start = first + i as u64 * page;
            let stop = start + page;
            stop.min(end).saturating_sub(start.max(offset))
        })
        .sum()
}

/// The nearest-rank percentile `q` in (0, 1] of `values`, or 0 when empty.
pub fn percentile(values: &[u64], q: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// A seeded order of every `(layer, expert)` pair (Fisher-Yates with a 64-bit LCG).
fn order(layers: usize, experts: u32, seed: u64) -> Vec<(usize, u32)> {
    let mut all: Vec<(usize, u32)> = (0..layers)
        .flat_map(|l| (0..experts).map(move |e| (l, e)))
        .collect();
    let mut state = seed;
    for i in (1..all.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = ((state >> 33) % (i as u64 + 1)) as usize;
        all.swap(i, j);
    }
    all
}

/// Which experts a case reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Set {
    /// The set that every `F_NOCACHE` case reads, so that the depths compare on the same byte
    /// ranges. Such a read leaves the file cache unchanged.
    Fixed,
    /// A set for this case alone. A buffered read fills the file cache, so the same experts are
    /// not cold again, and no cache purge is allowed.
    Fresh,
}

/// The cold cases in run order as `(mode, depth, set)`: the `F_NOCACHE` cases first (macOS), so
/// that no buffered read can warm the fixed set, then the buffered cases.
fn plan(nocache: bool, depths: &[usize]) -> Vec<(&'static str, usize, Set)> {
    let mut cases = Vec::new();
    if nocache {
        cases.extend(depths.iter().map(|&depth| ("nocache", depth, Set::Fixed)));
    }
    cases.extend(depths.iter().map(|&depth| ("buffered", depth, Set::Fresh)));
    cases
}

/// The experts of a case: the fixed set is drawn on first use and reused after that, and a fresh
/// set is drawn every time.
fn experts_for(
    set: Set,
    fixed: &mut Option<Vec<(usize, u32)>>,
    mut draw: impl FnMut() -> Result<Vec<(usize, u32)>, String>,
) -> Result<Vec<(usize, u32)>, String> {
    match (set, fixed.as_ref()) {
        (Set::Fixed, Some(list)) => Ok(list.clone()),
        (Set::Fixed, None) => {
            let list = draw()?;
            *fixed = Some(list.clone());
            Ok(list)
        }
        (Set::Fresh, _) => draw(),
    }
}

/// The `meta.txt` line that lists the fixed set as `layer:expert`, or `none`.
fn fixed_set_line(fixed: Option<&Vec<(usize, u32)>>) -> String {
    let list = fixed.map_or("none".to_string(), |list| {
        list.iter()
            .map(|(l, e)| format!("{l}:{e}"))
            .collect::<Vec<_>>()
            .join(",")
    });
    format!("fixed_set_layer_expert={list}\n")
}

/// Opened shards: buffered files, `F_NOCACHE` files (macOS), and residency maps.
struct Shards {
    files: Vec<File>,
    nocache: Option<Vec<File>>,
    maps: Vec<Residency>,
}

/// Reads the index and every shard header that holds an expert stack.
fn open(dir: &Path, prefix: &str, page: u64) -> Result<(ExpertLayout, Shards), String> {
    let index_path = dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path).map_err(|e| format!("{}: {e}", index_path.display()))?,
    )
    .map_err(|e| format!("index: {e}"))?;
    let names: BTreeSet<String> = index["weight_map"]
        .as_object()
        .ok_or("the index has no weight_map")?
        .iter()
        .filter(|(k, _)| k.starts_with(prefix) && k.contains(".mlp.switch_mlp."))
        .filter_map(|(_, v)| v.as_str().map(str::to_string))
        .collect();
    let mut headers = BTreeMap::new();
    for name in &names {
        let path = dir.join(name);
        let mut file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        let header = ShardHeader::read(&mut file, len, HEADER_LIMIT)
            .map_err(|e| format!("{name}: {e:?}"))?;
        headers.insert(name.clone(), header);
    }
    let layout = ExpertLayout::new(&index, &headers, prefix)?;
    let mut files = Vec::new();
    let mut nocache = Some(Vec::new());
    let mut maps = Vec::new();
    for name in &layout.shards {
        let path = dir.join(name);
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        maps.push(Residency::new(&file, page)?);
        files.push(file);
        nocache = match (nocache, sys::open_nocache(&path)?) {
            (Some(mut list), Some(file)) => {
                list.push(file);
                Some(list)
            }
            _ => None,
        };
    }
    Ok((
        layout,
        Shards {
            files,
            nocache,
            maps,
        },
    ))
}

/// Resident bytes of one expert.
fn resident(layout: &ExpertLayout, shards: &Shards, (l, e): (usize, u32)) -> Result<u64, String> {
    let mut sum = 0;
    for range in layout.ranges(l, e) {
        sum += shards.maps[range.shard].resident_bytes(range.offset, range.len)?;
    }
    Ok(sum)
}

/// One expert read.
struct Read {
    expert: (usize, u32),
    group: usize,
    bytes: u64,
    nanos: u64,
    cold_before: f64,
    resident_after: f64,
}

/// Reads `experts` in groups of `depth` threads through `files`. Returns the reads and the summed
/// wall time of the groups. Each group starts when all its threads pass one barrier.
fn read_case(
    layout: &ExpertLayout,
    shards: &Shards,
    files: &[File],
    experts: &[(usize, u32)],
    depth: usize,
    buffers: &mut [Vec<u8>],
) -> Result<(Vec<Read>, u64), String> {
    let mut reads = Vec::new();
    let mut wall = 0u64;
    for (group, chunk) in experts.chunks(depth).enumerate() {
        let mut before = Vec::new();
        for &expert in chunk {
            let bytes = layout.expert_bytes(expert.0);
            before.push((bytes, resident(layout, shards, expert)?));
        }
        let barrier = Barrier::new(chunk.len() + 1);
        let (nanos, elapsed) = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .zip(buffers.iter_mut())
                .map(|(&(l, e), buffer)| {
                    let barrier = &barrier;
                    scope.spawn(move || -> Result<u64, String> {
                        barrier.wait();
                        let started = Instant::now();
                        for range in layout.ranges(l, e) {
                            let len = usize::try_from(range.len).map_err(|_| "range length")?;
                            files[range.shard]
                                .read_exact_at(&mut buffer[..len], range.offset)
                                .map_err(|err| format!("read layer {l} expert {e}: {err}"))?;
                        }
                        Ok(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
                    })
                })
                .collect();
            barrier.wait();
            let started = Instant::now();
            let nanos: Vec<Result<u64, String>> = handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("reader panicked".into())))
                .collect();
            (nanos, started.elapsed())
        });
        wall += u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        for ((&expert, (bytes, resident_before)), nanos) in chunk.iter().zip(before).zip(nanos) {
            let after = resident(layout, shards, expert)?;
            reads.push(Read {
                expert,
                group,
                bytes,
                nanos: nanos?,
                cold_before: 1.0 - resident_before as f64 / bytes as f64,
                resident_after: after as f64 / bytes as f64,
            });
        }
    }
    Ok((reads, wall))
}

/// Draws the next `count` experts of `candidates` whose cold share (`cold_of`) is at least
/// `min_cold`. A candidate below it counts in `skipped`.
fn draw(
    candidates: &mut impl Iterator<Item = (usize, u32)>,
    count: usize,
    min_cold: f64,
    skipped: &mut usize,
    mut cold_of: impl FnMut((usize, u32)) -> Result<f64, String>,
) -> Result<Vec<(usize, u32)>, String> {
    let mut out = Vec::new();
    while out.len() < count {
        let expert = candidates
            .next()
            .ok_or("ran out of cold experts: the file cache holds most of the checkpoint")?;
        if cold_of(expert)? >= min_cold {
            out.push(expert);
        } else {
            *skipped += 1;
        }
    }
    Ok(out)
}

/// Rates of one case.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CaseStats {
    bytes_per_second: f64,
    /// Cold share before the reads, weighted by bytes.
    cold: f64,
    /// Resident share after the reads, weighted by bytes.
    after: f64,
}

/// The rates of `reads` that took `wall` nanoseconds together.
fn case_stats(reads: &[Read], wall: u64) -> CaseStats {
    let bytes: u64 = reads.iter().map(|r| r.bytes).sum();
    let weighted = |f: fn(&Read) -> f64| {
        reads.iter().map(|r| f(r) * r.bytes as f64).sum::<f64>() / bytes.max(1) as f64
    };
    CaseStats {
        bytes_per_second: bytes as f64 / (wall.max(1) as f64 / 1e9),
        cold: weighted(|r| r.cold_before),
        after: weighted(|r| r.resident_after),
    }
}

/// One output line of `summary.tsv`.
fn summarize(case: &str, depth: usize, reads: &[Read], wall: u64, skipped: usize) -> String {
    let bytes: u64 = reads.iter().map(|r| r.bytes).sum();
    let nanos: Vec<u64> = reads.iter().map(|r| r.nanos).collect();
    let stats = case_stats(reads, wall);
    format!(
        "{case}\t{depth}\t{}\t{bytes}\t{wall}\t{:.0}\t{}\t{}\t{:.4}\t{:.4}\t{skipped}\t{:.3}\n",
        reads.len(),
        stats.bytes_per_second,
        percentile(&nanos, 0.5),
        percentile(&nanos, 0.95),
        stats.cold,
        stats.after,
        // Gigabytes (10^9 bytes) per second.
        stats.bytes_per_second / 1e9,
    )
}

/// One cold case of a read mode: depth, bytes per second, and cold share.
type ColdCase = (usize, f64, f64);

/// The verdict lines of `meta.txt`: for each mode the depth-1 rate, the best depth and rate, and
/// the smallest cold share; then whether `mincore` saw the file cache (every buffered cold case
/// left at least `min_cold` of its bytes resident) and T0-P4. T0-P4 passes when `mincore` saw the
/// cache and every mode has a depth-1 case and a cold share of at least `min_cold` in every case.
fn verdict(
    cold_rates: &BTreeMap<&str, Vec<ColdCase>>,
    buffered_after: &[f64],
    min_cold: f64,
) -> String {
    let resident_after_ok = buffered_after.iter().all(|&after| after >= min_cold);
    let mut verdict = String::new();
    let mut pass = resident_after_ok;
    for (mode, rates) in cold_rates {
        let qd1 = rates.iter().find(|r| r.0 == 1);
        let best = rates.iter().max_by(|a, b| a.1.total_cmp(&b.1));
        let lowest_cold = rates.iter().map(|r| r.2).fold(f64::INFINITY, f64::min);
        pass &= qd1.is_some() && lowest_cold >= min_cold;
        let _ = writeln!(
            verdict,
            "{mode}_qd1_bytes_per_second={:.0}\n{mode}_best_depth={}\n{mode}_best_bytes_per_second={:.0}\n\
             {mode}_min_cold_fraction={lowest_cold:.4}",
            qd1.map_or(0.0, |r| r.1),
            best.map_or(0, |r| r.0),
            best.map_or(0.0, |r| r.1),
        );
        for &(depth, rate, _) in rates {
            let _ = writeln!(
                verdict,
                "{mode}_gb_per_second_depth{depth}={:.3}",
                rate / 1e9
            );
        }
    }
    let _ = writeln!(
        verdict,
        "mincore_sees_file_cache={resident_after_ok}\nt0_p4={}",
        if pass { "pass" } else { "fail" }
    );
    verdict
}

/// Output of a command, or `unavailable`.
fn command(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or("unavailable".to_string(), |o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .replace('\n', " | ")
        })
}

fn run(settings: &Settings) -> Result<(), String> {
    let page = sys::page_size()?;
    let (layout, shards) = open(&settings.artifact, &settings.prefix, page)?;
    std::fs::create_dir_all(&settings.out).map_err(|e| e.to_string())?;
    let unix = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    };
    let mut meta = String::new();
    let _ = write!(
        meta,
        "schema=s3-expert-reads/1\nartifact={}\nos={}\npage_bytes={page}\nlayers={}\n\
         experts={}\nexpert_bytes_layer0={}\nseed={}\nexperts_per_case={}\ndepths={:?}\n\
         min_cold={}\nnocache={}\nstart_unix={}\nswap_before={}\nvm_stat_before={}\n",
        settings.artifact.display(),
        std::env::consts::OS,
        layout.stacks.len(),
        layout.experts,
        layout.expert_bytes(0),
        settings.seed,
        settings.experts,
        settings.depths,
        settings.min_cold,
        if shards.nocache.is_some() {
            "F_NOCACHE"
        } else {
            "unsupported"
        },
        unix(),
        command("sysctl", &["-n", "vm.swapusage"]),
        command("vm_stat", &[]),
    );
    let max_bytes = (0..layout.stacks.len())
        .map(|l| layout.expert_bytes(l))
        .max()
        .unwrap_or(0);
    let max_depth = settings.depths.iter().copied().max().unwrap_or(1);
    let mut buffers: Vec<Vec<u8>> = (0..max_depth)
        .map(|_| vec![1u8; usize::try_from(max_bytes).unwrap_or(0)])
        .collect();
    let mut candidates = order(layout.stacks.len(), layout.experts, settings.seed).into_iter();
    let mut reads_tsv =
        String::from("case\tdepth\tgroup\tlayer\texpert\tbytes\tns\tcold_before\tresident_after\n");
    let mut summary = String::from(
        "case\tdepth\texperts\tbytes\twall_ns\tbytes_per_second\tp50_ns\tp95_ns\tcold_fraction\t\
         resident_after_fraction\tskipped\tgb_per_second\n",
    );
    let mut cold_rates: BTreeMap<&str, Vec<ColdCase>> = BTreeMap::new();
    let mut buffered_after = Vec::new();
    let mut record = |case: String, depth: usize, reads: &[Read], wall: u64, skipped: usize| {
        for r in reads {
            let _ = writeln!(
                reads_tsv,
                "{case}\t{depth}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                r.group, r.expert.0, r.expert.1, r.bytes, r.nanos, r.cold_before, r.resident_after
            );
        }
        let line = summarize(&case, depth, reads, wall, skipped);
        summary.push_str(&line);
        line
    };
    let mut fixed: Option<Vec<(usize, u32)>> = None;
    for (mode, depth, set) in plan(shards.nocache.is_some(), &settings.depths) {
        let files: &[File] = match (&shards.nocache, mode) {
            (Some(nocache), "nocache") => nocache,
            _ => &shards.files,
        };
        let mut skipped = 0;
        let experts = experts_for(set, &mut fixed, || {
            draw(
                &mut candidates,
                settings.experts,
                settings.min_cold,
                &mut skipped,
                |expert| {
                    let bytes = layout.expert_bytes(expert.0) as f64;
                    Ok(1.0 - resident(&layout, &shards, expert)? as f64 / bytes)
                },
            )
        })?;
        let (reads, wall) = read_case(&layout, &shards, files, &experts, depth, &mut buffers)?;
        let stats = case_stats(&reads, wall);
        cold_rates
            .entry(mode)
            .or_default()
            .push((depth, stats.bytes_per_second, stats.cold));
        print!(
            "{}",
            record(format!("cold-{mode}"), depth, &reads, wall, skipped)
        );
        if mode == "buffered" {
            buffered_after.push(stats.after);
            let (warm, warm_wall) =
                read_case(&layout, &shards, files, &experts, depth, &mut buffers)?;
            print!(
                "{}",
                record(format!("warm-{mode}"), depth, &warm, warm_wall, 0)
            );
        }
    }
    let verdict = format!(
        "{}{}",
        verdict(&cold_rates, &buffered_after, settings.min_cold),
        fixed_set_line(fixed.as_ref())
    );
    let _ = write!(
        meta,
        "end_unix={}\nswap_after={}\nvm_stat_after={}\n{verdict}",
        unix(),
        command("sysctl", &["-n", "vm.swapusage"]),
        command("vm_stat", &[]),
    );
    for (name, body) in [
        ("reads.tsv", &reads_tsv),
        ("summary.tsv", &summary),
        ("meta.txt", &meta),
    ] {
        std::fs::write(settings.out.join(name), body).map_err(|e| format!("{name}: {e}"))?;
    }
    print!("{verdict}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_bytes_count_the_overlap_of_resident_pages() {
        // Bytes 1,000 to 10,000 over 4,096-byte pages 0, 1, and 2; pages 0 and 2 are resident:
        // 4,096 - 1,000 = 3,096 bytes of page 0 and 10,000 - 8,192 = 1,808 bytes of page 2.
        assert_eq!(resident_bytes(&[1, 0, 1], 4096, 0, 1000, 9000), 4904);
        // Only the low bit means resident (macOS also sets reference and modify bits).
        assert_eq!(resident_bytes(&[2, 0, 0], 4096, 0, 1000, 9000), 0);
    }

    #[test]
    fn the_order_holds_every_pair_once() {
        let all = order(3, 5, 7);
        let set: BTreeSet<(usize, u32)> = all.iter().copied().collect();
        assert_eq!((all.len(), set.len()), (15, 15), "15 distinct pairs");
        assert_ne!(all, order(3, 5, 8), "another seed gives another order");
    }

    #[test]
    fn every_case_reads_its_experts_from_a_synthetic_checkpoint() {
        let dir = warm_checkpoint("cases");
        // The freshly written shard is in the file cache, so every expert is warm: a cold share
        // of 0 admits them all.
        let settings = Settings {
            artifact: dir.clone(),
            out: dir.join("out"),
            experts: 4,
            depths: vec![1, 2],
            seed: 3,
            min_cold: 0.0,
            prefix: "m.".to_string(),
        };
        assert_eq!(run(&settings), Ok(()));
        let summary = std::fs::read_to_string(dir.join("out/summary.tsv")).unwrap();
        // 4 experts of 360 bytes per case.
        for case in [
            "cold-buffered\t1\t4\t1440\t",
            "warm-buffered\t1\t4\t1440\t",
            "cold-buffered\t2\t4\t1440\t",
            "warm-buffered\t2\t4\t1440\t",
        ] {
            assert!(
                summary.lines().any(|line| line.starts_with(case)),
                "{case:?} missing from {summary}"
            );
        }
        let reads = std::fs::read_to_string(dir.join("out/reads.tsv")).unwrap();
        assert_eq!(cold_set_violation(&reads), None, "{reads}");
        let meta = std::fs::read_to_string(dir.join("out/meta.txt")).unwrap();
        // The host decides whether `F_NOCACHE` cases ran. The files must agree with each other, and
        // the run must not depend on which way it went.
        let nocache_ran = !meta.contains("nocache=unsupported");
        assert_eq!(
            summary
                .lines()
                .any(|line| line.starts_with("cold-nocache\t1\t4\t1440\t")),
            nocache_ran,
            "nocache rows against nocache support in {meta}"
        );
        let meta = std::fs::read_to_string(dir.join("out/meta.txt")).unwrap();
        assert!(
            meta.contains("mincore_sees_file_cache=true") && meta.contains("t0_p4=pass"),
            "{meta}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The first broken rule in a `reads.tsv`, or `None`. A buffered case reads a set of its own
    /// (no expert is read cold twice, and a set is disjoint from the fixed set). Every
    /// `F_NOCACHE` case reads exactly one fixed set, so the depths compare on the same byte
    /// ranges. The rule does not look at cold shares, because residency is reported, not assumed.
    fn cold_set_violation(reads: &str) -> Option<String> {
        // Cold rows as (mode, depth, layer:expert).
        let rows: Vec<(&str, &str, String)> = reads
            .lines()
            .skip(1)
            .filter_map(|line| {
                let f: Vec<&str> = line.split('\t').collect();
                let mode = f.first()?.strip_prefix("cold-")?;
                Some((mode, *f.get(1)?, format!("{}:{}", f.get(3)?, f.get(4)?)))
            })
            .collect();
        let buffered: Vec<&String> = rows
            .iter()
            .filter(|r| r.0 == "buffered")
            .map(|r| &r.2)
            .collect();
        let distinct: BTreeSet<&String> = buffered.iter().copied().collect();
        if distinct.len() != buffered.len() {
            return Some("a buffered case read an expert cold twice".to_string());
        }
        let mut per_depth: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.0 == "nocache") {
            per_depth.entry(r.1).or_default().push(&r.2);
        }
        let sets: BTreeSet<BTreeSet<&String>> = per_depth
            .values()
            .map(|list| list.iter().copied().collect())
            .collect();
        if sets.len() > 1 {
            return Some("the nocache depths read different sets".to_string());
        }
        if per_depth
            .values()
            .any(|list| list.iter().collect::<BTreeSet<_>>().len() != list.len())
        {
            return Some("a nocache case read an expert twice".to_string());
        }
        if sets.iter().flatten().any(|e| distinct.contains(e)) {
            return Some("the fixed set and a buffered set share an expert".to_string());
        }
        None
    }

    #[test]
    fn the_cold_set_rule_holds_with_and_without_nocache_cases() {
        let row = |case: &str, depth: u32, expert: u32| {
            format!("{case}\t{depth}\t0\t0\t{expert}\t9\t9\t0.0000\t0.0000\n")
        };
        let header = "case\tdepth\tgroup\tlayer\texpert\tbytes\tns\tcold_before\tresident_after\n";
        // Linux: buffered cases only, each with its own experts; the warm re-read is ignored.
        let buffered_only = format!(
            "{header}{}{}{}",
            row("cold-buffered", 1, 0),
            row("warm-buffered", 1, 0),
            row("cold-buffered", 2, 1)
        );
        assert_eq!(cold_set_violation(&buffered_only), None);
        // macOS: two nocache depths re-read one fixed set, which no buffered case shares.
        let with_nocache = format!(
            "{header}{}{}{}{}{}{}",
            row("cold-nocache", 1, 5),
            row("cold-nocache", 1, 6),
            row("cold-nocache", 2, 6),
            row("cold-nocache", 2, 5),
            row("cold-buffered", 1, 0),
            row("cold-buffered", 2, 1)
        );
        assert_eq!(cold_set_violation(&with_nocache), None);
        // Each branch still catches its own breach.
        let repeated = format!("{buffered_only}{}", row("cold-buffered", 4, 0));
        assert!(cold_set_violation(&repeated).is_some_and(|m| m.contains("cold twice")));
        let drifting = format!("{with_nocache}{}", row("cold-nocache", 4, 7));
        assert!(cold_set_violation(&drifting).is_some_and(|m| m.contains("different sets")));
        let shared = format!("{with_nocache}{}", row("cold-buffered", 4, 5));
        assert!(cold_set_violation(&shared).is_some_and(|m| m.contains("share an expert")));
    }

    /// Whether the `fixed_set_layer_expert` line of a `meta.txt` fits its `nocache` line: `none`
    /// without `F_NOCACHE`, and `experts` pairs `layer:expert` with it.
    fn fixed_set_fits_nocache(meta: &str, experts: usize) -> bool {
        let Some(list) = meta
            .lines()
            .find_map(|line| line.strip_prefix("fixed_set_layer_expert="))
        else {
            return false;
        };
        if meta.contains("nocache=unsupported") {
            list == "none"
        } else {
            list.split(',').count() == experts && list.split(',').all(|p| p.contains(':'))
        }
    }

    #[test]
    fn the_fixed_set_line_fits_the_nocache_support_on_both_paths() {
        let linux = "nocache=unsupported\nfixed_set_layer_expert=none\n";
        let mac = "nocache=F_NOCACHE\nfixed_set_layer_expert=0:9,0:12\n";
        assert!(fixed_set_fits_nocache(linux, 2) && fixed_set_fits_nocache(mac, 2));
        assert!(!fixed_set_fits_nocache(mac, 3), "wrong pair count");
        let mac_none = "nocache=F_NOCACHE\nfixed_set_layer_expert=none\n";
        assert!(!fixed_set_fits_nocache(mac_none, 2));
        let linux_set = "nocache=unsupported\nfixed_set_layer_expert=0:9,0:12\n";
        assert!(!fixed_set_fits_nocache(linux_set, 2));
        assert!(
            !fixed_set_fits_nocache("nocache=unsupported\n", 2),
            "line missing"
        );
    }

    #[test]
    fn percentiles_use_the_nearest_rank() {
        assert_eq!(percentile(&[50, 10, 40, 20, 30], 0.5), 30);
        assert_eq!(percentile(&[50, 10, 40, 20, 30], 0.95), 50);
    }

    /// A synthetic checkpoint in a fresh directory. The freshly written shard sits in the file
    /// cache, so every expert is warm.
    fn warm_checkpoint(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("expert-reads-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (file, index) = layout::synthetic_shard(16);
        std::fs::write(dir.join("s.safetensors"), file).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn a_warm_checkpoint_has_no_cold_experts_at_the_default_cold_share() {
        // The shard was just written, so `mincore` reports every page resident: no expert is 0.9
        // cold, every candidate is skipped, and the draw runs out (16 experts, 1 layer).
        let dir = warm_checkpoint("warm");
        let settings = Settings {
            artifact: dir.clone(),
            out: dir.join("out"),
            experts: 4,
            depths: vec![1],
            seed: 3,
            min_cold: 0.9,
            prefix: "m.".to_string(),
        };
        let result = run(&settings);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            result
                .as_ref()
                .is_err_and(|m| m.contains("ran out of cold experts")),
            "{result:?}"
        );
    }

    #[test]
    fn the_draw_keeps_the_experts_at_least_as_cold_as_asked_and_counts_the_rest() {
        // Candidates in order with cold shares 1.0, 0.5, 0.95, 0.8, 0.9, 1.0. At least 0.9 cold
        // admits experts 0, 2, and 4 (0.9 itself counts) and skips 1 and 3 on the way.
        let cold = [1.0, 0.5, 0.95, 0.8, 0.9, 1.0];
        let mut candidates = (0..6u32).map(|e| (0usize, e));
        let mut skipped = 0;
        let drawn = draw(&mut candidates, 3, 0.9, &mut skipped, |(_, e)| {
            Ok(cold[e as usize])
        });
        assert_eq!(drawn, Ok(vec![(0, 0), (0, 2), (0, 4)]));
        assert_eq!(skipped, 2);
        // The next draw continues after expert 4: expert 5 is cold, and then none is left.
        let next = draw(&mut candidates, 2, 0.9, &mut skipped, |(_, e)| {
            Ok(cold[e as usize])
        });
        assert!(
            next.is_err_and(|m| m.contains("ran out of cold experts")),
            "the draw needs a second cold expert"
        );
    }

    #[test]
    fn the_draw_passes_on_a_failed_residency_check() {
        let mut candidates = std::iter::once((0usize, 0u32));
        let drawn = draw(&mut candidates, 1, 0.9, &mut 0, |_| {
            Err("mincore".to_string())
        });
        assert_eq!(drawn, Err("mincore".to_string()));
    }

    /// Reads of 1,000 bytes (1 ms, cold share 1.0, resident after 1.0) and 3,000 bytes (3 ms,
    /// cold share 0.6, resident after 0.8).
    fn two_reads() -> Vec<Read> {
        vec![
            Read {
                expert: (0, 1),
                group: 0,
                bytes: 1000,
                nanos: 3_000_000,
                cold_before: 1.0,
                resident_after: 1.0,
            },
            Read {
                expert: (0, 2),
                group: 0,
                bytes: 3000,
                nanos: 1_000_000,
                cold_before: 0.6,
                resident_after: 0.8,
            },
        ]
    }

    #[test]
    fn a_summary_line_reports_bytes_per_second_and_byte_weighted_shares() {
        // 4,000 bytes in a group wall time of 4,000,000 ns = 4 ms: 4,000 / 0.004 = 1,000,000
        // bytes per second. Latencies sorted: 1,000,000 and 3,000,000; the nearest-rank p50 is
        // rank ceil(0.5 * 2) = 1 and p95 is rank ceil(1.9) = 2. Cold share:
        // (1000 * 1.0 + 3000 * 0.6) / 4000 = 2800 / 4000 = 0.7. Resident after:
        // (1000 * 1.0 + 3000 * 0.8) / 4000 = 3400 / 4000 = 0.85. The last field is
        // 1,000,000 / 10^9 = 0.001 GB per second.
        assert_eq!(
            summarize("cold-buffered", 2, &two_reads(), 4_000_000, 5),
            "cold-buffered\t2\t2\t4000\t4000000\t1000000\t1000000\t3000000\t0.7000\t0.8500\t5\t0.001\n"
        );
        let stats = case_stats(&two_reads(), 4_000_000);
        assert!((stats.bytes_per_second - 1e6).abs() < 1e-6, "{stats:?}");
        assert!((stats.cold - 0.7).abs() < 1e-12, "{stats:?}");
        assert!((stats.after - 0.85).abs() < 1e-12, "{stats:?}");
    }

    /// Two modes with the cases `(depth, bytes per second, cold share)`.
    fn passing_rates() -> BTreeMap<&'static str, Vec<ColdCase>> {
        BTreeMap::from([
            (
                "buffered",
                vec![
                    (1, 1e8, 0.97),
                    (2, 2e8, 0.95),
                    (4, 4e8, 0.99),
                    (8, 3e8, 0.92),
                ],
            ),
            ("nocache", vec![(1, 5e7, 0.96), (8, 2e8, 0.91)]),
        ])
    }

    #[test]
    fn the_verdict_names_the_depth_one_and_best_rates_and_passes() {
        // Buffered: depth 1 has 1e8, the best rate is 4e8 at depth 4, the smallest cold share is
        // 0.92. No-cache: depth 1 has 5e7, the best is 2e8 at depth 8, the smallest cold share is
        // 0.91. Every share is at least 0.9, depth 1 exists in both modes, and the buffered reads
        // left at least 0.9 resident (0.98 is the lowest), so T0-P4 passes. Each depth also gets
        // its rate in GB/s: 1e8 B/s = 0.100, 2e8 = 0.200, 4e8 = 0.400, 3e8 = 0.300 (buffered) and
        // 5e7 = 0.050, 2e8 = 0.200 (no-cache).
        let text = verdict(&passing_rates(), &[0.99, 1.0, 1.0, 0.98], 0.9);
        assert_eq!(
            text,
            "buffered_qd1_bytes_per_second=100000000\n\
             buffered_best_depth=4\n\
             buffered_best_bytes_per_second=400000000\n\
             buffered_min_cold_fraction=0.9200\n\
             buffered_gb_per_second_depth1=0.100\n\
             buffered_gb_per_second_depth2=0.200\n\
             buffered_gb_per_second_depth4=0.400\n\
             buffered_gb_per_second_depth8=0.300\n\
             nocache_qd1_bytes_per_second=50000000\n\
             nocache_best_depth=8\n\
             nocache_best_bytes_per_second=200000000\n\
             nocache_min_cold_fraction=0.9100\n\
             nocache_gb_per_second_depth1=0.050\n\
             nocache_gb_per_second_depth8=0.200\n\
             mincore_sees_file_cache=true\n\
             t0_p4=pass\n"
        );
    }

    #[test]
    fn the_verdict_fails_when_a_case_is_less_cold_than_asked() {
        // The depth-8 buffered case has cold share 0.89 < 0.9.
        let mut rates = passing_rates();
        rates.get_mut("buffered").unwrap()[3].2 = 0.89;
        let text = verdict(&rates, &[1.0; 4], 0.9);
        assert!(
            text.contains("buffered_min_cold_fraction=0.8900\n")
                && text.contains("mincore_sees_file_cache=true\n")
                && text.ends_with("t0_p4=fail\n"),
            "{text}"
        );
    }

    #[test]
    fn the_verdict_accepts_a_case_that_is_exactly_as_cold_as_asked() {
        let rates = BTreeMap::from([("buffered", vec![(1, 1e8, 0.9)])]);
        assert!(verdict(&rates, &[1.0], 0.9).ends_with("t0_p4=pass\n"));
    }

    #[test]
    fn the_verdict_fails_without_a_depth_one_case() {
        // Depths 2 and 4 only: the depth-1 rate is 0 and the best is depth 4.
        let rates = BTreeMap::from([("buffered", vec![(2, 2e8, 0.95), (4, 4e8, 0.96)])]);
        let text = verdict(&rates, &[1.0, 1.0], 0.9);
        assert!(
            text.contains("buffered_qd1_bytes_per_second=0\n")
                && text.contains("buffered_best_depth=4\n")
                && text.ends_with("t0_p4=fail\n"),
            "{text}"
        );
    }

    #[test]
    fn the_verdict_fails_when_mincore_does_not_see_the_file_cache() {
        // One buffered case left only 0.5 of its bytes resident after the read.
        let text = verdict(&passing_rates(), &[1.0, 0.5, 1.0, 1.0], 0.9);
        assert!(
            text.contains("mincore_sees_file_cache=false\n") && text.ends_with("t0_p4=fail\n"),
            "{text}"
        );
    }

    #[test]
    fn the_defaults_cover_queue_depths_1_to_16_with_64_experts() {
        let settings = parse(vec![
            "--artifact".to_string(),
            "a".to_string(),
            "--out".to_string(),
            "o".to_string(),
        ])
        .unwrap();
        assert_eq!(settings.depths, vec![1, 2, 4, 8, 16]);
        assert_eq!(settings.experts, 64);
    }

    #[test]
    fn nocache_cases_share_one_set_and_buffered_cases_take_their_own() {
        // F_NOCACHE leaves the file cache unchanged, so its depths can read the same set, and it
        // runs first so that no buffered read warms that set. A buffered read fills the cache.
        assert_eq!(
            plan(true, &[1, 2]),
            vec![
                ("nocache", 1, Set::Fixed),
                ("nocache", 2, Set::Fixed),
                ("buffered", 1, Set::Fresh),
                ("buffered", 2, Set::Fresh),
            ]
        );
        assert_eq!(plan(false, &[4]), vec![("buffered", 4, Set::Fresh)]);
    }

    #[test]
    fn the_fixed_set_is_drawn_once_and_each_fresh_set_is_drawn_anew() {
        let mut calls = 0u32;
        let mut fixed = None;
        let mut next = || {
            calls += 1;
            Ok(vec![(0usize, calls)])
        };
        // The draws return (0, 1), (0, 2), and (0, 3) in order. The fixed set takes the first
        // draw and keeps it, and each fresh set takes the next draw.
        let first = experts_for(Set::Fixed, &mut fixed, &mut next).unwrap();
        let again = experts_for(Set::Fixed, &mut fixed, &mut next).unwrap();
        let fresh = experts_for(Set::Fresh, &mut fixed, &mut next).unwrap();
        let other = experts_for(Set::Fresh, &mut fixed, &mut next).unwrap();
        assert_eq!(
            (first, again, fresh, other),
            (vec![(0, 1)], vec![(0, 1)], vec![(0, 2)], vec![(0, 3)])
        );
    }

    #[test]
    fn the_fixed_set_line_lists_layer_and_expert() {
        assert_eq!(fixed_set_line(None), "fixed_set_layer_expert=none\n");
        assert_eq!(
            fixed_set_line(Some(&vec![(0, 3), (12, 407)])),
            "fixed_set_layer_expert=0:3,12:407\n"
        );
    }

    #[test]
    fn the_summary_reports_gigabytes_per_second() {
        // Two experts of 1,000,000,000 bytes in a group wall time of 1,000,000,000 ns (1 s):
        // 2,000,000,000 bytes per second, which is 2.000 GB/s (10^9 bytes).
        let read = |group| Read {
            expert: (0, group as u32),
            group,
            bytes: 1_000_000_000,
            nanos: 1_000_000_000,
            cold_before: 1.0,
            resident_after: 1.0,
        };
        let line = summarize("cold-buffered", 2, &[read(0), read(0)], 1_000_000_000, 0);
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        assert_eq!(fields[5], "2000000000", "bytes per second in {line}");
        assert_eq!(fields.last(), Some(&"2.000"), "GB per second in {line}");
    }

    #[test]
    fn every_default_depth_reports_its_rate_in_gigabytes_per_second() {
        let dir = warm_checkpoint("depths");
        // 5 depths of 2 experts each draw 10 of the 16 experts. A cold share of 0 admits the warm
        // experts.
        let settings = Settings {
            artifact: dir.clone(),
            out: dir.join("out"),
            experts: 2,
            depths: vec![1, 2, 4, 8, 16],
            seed: 3,
            min_cold: 0.0,
            prefix: "m.".to_string(),
        };
        assert_eq!(run(&settings), Ok(()));
        let meta = std::fs::read_to_string(dir.join("out/meta.txt")).unwrap();
        for depth in [1, 2, 4, 8, 16] {
            let key = format!("buffered_gb_per_second_depth{depth}=");
            assert!(meta.contains(&key), "{key} missing from {meta}");
        }
        // `none` without `F_NOCACHE`, and the fixed set of 2 experts with it.
        assert!(fixed_set_fits_nocache(&meta, 2), "{meta}");
        // The depth-16 row ends with its rate: 720 bytes over the group wall time.
        let summary = std::fs::read_to_string(dir.join("out/summary.tsv")).unwrap();
        let row = summary
            .lines()
            .find(|line| line.starts_with("cold-buffered\t16\t2\t720\t"))
            .unwrap();
        let fields: Vec<&str> = row.split('\t').collect();
        let rate: f64 = fields[5].parse().unwrap();
        let gb: f64 = fields[fields.len() - 1].parse().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            (gb - rate / 1e9).abs() <= 0.001,
            "{gb} GB/s against {rate} B/s"
        );
    }
}
