//! The replay grid, its aggregates, the T0 checks, and the decision value (S3 plan sections 3.5
//! and 3.7). Every replay is held out: the ranking that sets static cores and warm starts comes
//! from the other prompts.

use crate::cache::{Kind, POLICIES, Policy};
use crate::sim::{COLD, Outcome, SimConfig, greedy_split, ranking, simulate, stack_distances};
use crate::trace::{PromptTrace, TokenLog, Trace, repetition};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache sizes as percentages of the experts of a layer (plan section 3.5, "Sizes").
pub const PERCENTS: [usize; 9] = [10, 15, 20, 25, 30, 35, 40, 45, 50];
/// Page-cache sizes of the second level in GiB.
pub const PAGE_CACHE_GIB: [u64; 3] = [0, 2, 4];
/// Decode windows from the start of each prompt (the cold-trial question).
pub const WINDOWS: [usize; 3] = [16, 64, 256];

/// A named miss bandwidth in bytes per second.
#[derive(Debug, Clone, PartialEq)]
pub struct Bandwidth {
    pub name: String,
    pub bytes_per_second: f64,
}

/// Inputs of a report.
#[derive(Debug, Clone)]
pub struct Settings {
    pub planning_slots: usize,
    /// Seconds per generated token without expert reads (Z1c profile P2).
    pub t_fixed: f64,
    pub bandwidths: Vec<Bandwidth>,
    /// Bandwidth of a page-cache hit.
    pub page_cache_bandwidth: f64,
    /// The bandwidth named here is `bw_cold` of the gates (T0b, best queue depth).
    pub cold: Option<String>,
    /// Prompts whose generated 4-gram repetition exceeds this share leave the decision value.
    pub repetition_limit: f64,
    pub threads: usize,
}

/// One replay setting over every held-out prompt.
#[derive(Debug, Clone)]
pub struct Job {
    pub policy: Policy,
    /// `uniform` or `greedy`.
    pub split: &'static str,
    /// `p10` to `p50` or `plan`.
    pub size: String,
    /// Slots per layer for each held-out prompt (a greedy split depends on the training prompts).
    pub slots: Vec<Vec<usize>>,
    pub warm: bool,
    pub admit_prompt: bool,
    pub page_cache_gib: u64,
}

/// The rate `hits / accesses`, or 0 for no accesses.
pub fn rate(hits: u64, accesses: u64) -> f64 {
    if accesses == 0 {
        0.0
    } else {
        hits as f64 / accesses as f64
    }
}

/// The nearest-rank percentile `q` in (0, 1] of `values`, or 0 when empty.
pub fn percentile(values: &[u32], q: f64) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Aggregates of a set of outcomes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Summary {
    pub decode_tokens: usize,
    pub decode_hits: u64,
    pub decode_accesses: u64,
    pub prompt_hits: u64,
    pub prompt_accesses: u64,
    /// Hits and accesses of the first `WINDOWS[i]` generated steps of each prompt.
    pub windows: [(u64, u64); 3],
    pub misses: Vec<u32>,
    pub page_cache_bytes: u64,
    pub ssd_bytes: u64,
    pub layer_hits: Vec<u64>,
    pub layer_accesses: Vec<u64>,
}

impl Summary {
    pub fn add(&mut self, outcome: &Outcome, step_accesses: u64) {
        self.decode_tokens += outcome.decode.len();
        for (i, step) in outcome.decode.iter().enumerate() {
            self.decode_hits += u64::from(step.hits);
            self.decode_accesses += step_accesses;
            for (w, &n) in WINDOWS.iter().enumerate() {
                if i < n {
                    self.windows[w].0 += u64::from(step.hits);
                    self.windows[w].1 += step_accesses;
                }
            }
            self.misses.push(step.misses);
            self.page_cache_bytes += step.page_cache_bytes;
            self.ssd_bytes += step.ssd_bytes;
        }
        self.prompt_hits += outcome.prompt_hits;
        self.prompt_accesses += outcome.prompt_accesses;
        if self.layer_hits.is_empty() {
            self.layer_hits = vec![0; outcome.layer_hits.len()];
            self.layer_accesses = vec![0; outcome.layer_accesses.len()];
        }
        for (sum, v) in self.layer_hits.iter_mut().zip(&outcome.layer_hits) {
            *sum += v;
        }
        for (sum, v) in self.layer_accesses.iter_mut().zip(&outcome.layer_accesses) {
            *sum += v;
        }
    }

    pub fn hit_rate(&self) -> f64 {
        rate(self.decode_hits, self.decode_accesses)
    }

    /// Mean bytes per generated token: page cache and SSD.
    pub fn bytes_per_token(&self) -> (f64, f64) {
        let n = self.decode_tokens.max(1) as f64;
        (self.page_cache_bytes as f64 / n, self.ssd_bytes as f64 / n)
    }
}

/// Read seconds per generated token: page-cache bytes at `page_cache_bw` and SSD bytes at `bw`.
pub fn read_seconds(summary: &Summary, bw: f64, page_cache_bw: f64) -> f64 {
    let (page_cache, ssd) = summary.bytes_per_token();
    page_cache / page_cache_bw + ssd / bw
}

/// The T0 gates (plan section 3.7) for decision value `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gates {
    /// `t_fixed + (1 - h) * miss_bytes / bw_cold`.
    pub g1_seconds: f64,
    pub g1: bool,
    /// `h * miss_bytes / bw_cold`.
    pub g2_saved: f64,
    pub g2: bool,
}

/// T0-G1: at most 0.200 s per token. T0-G2: at least 0.050 s saved and `h` at least 1.5 times
/// the capacity share.
pub fn gates(h: f64, t_fixed: f64, miss_bytes: f64, bw_cold: f64, share: f64) -> Gates {
    let g1_seconds = t_fixed + (1.0 - h) * miss_bytes / bw_cold;
    let g2_saved = h * miss_bytes / bw_cold;
    Gates {
        g1_seconds,
        g1: g1_seconds <= 0.200,
        g2_saved,
        g2: g2_saved >= 0.050 && h >= 1.5 * share,
    }
}

/// Runs `jobs` on every prompt of `trace` with `threads` workers. `rankings[i]` is the training
/// ranking of held-out prompt `i`.
pub fn run_jobs(
    trace: &Trace,
    rankings: &[Vec<Vec<u16>>],
    jobs: &[Job],
    threads: usize,
) -> Result<Vec<Vec<Outcome>>, String> {
    let tasks: Vec<(usize, usize)> = (0..jobs.len())
        .flat_map(|j| (0..trace.prompts.len()).map(move |p| (j, p)))
        .collect();
    let results: Mutex<Vec<Vec<Option<Outcome>>>> =
        Mutex::new(vec![vec![None; trace.prompts.len()]; jobs.len()]);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(j, p)) = tasks.get(index) else {
                        break;
                    };
                    let job = &jobs[j];
                    let config = SimConfig {
                        policy: job.policy,
                        slots: &job.slots[p],
                        warm: job.warm,
                        admit_prompt: job.admit_prompt,
                        page_cache_bytes: job.page_cache_gib << 30,
                    };
                    match simulate(trace, &trace.prompts[p], &rankings[p], &config) {
                        Ok(outcome) => {
                            if let Ok(mut all) = results.lock() {
                                all[j][p] = Some(outcome);
                            }
                        }
                        Err(message) => {
                            if let Ok(mut slot) = error.lock() {
                                slot.get_or_insert(format!("{}: {message}", job.policy.name));
                            }
                        }
                    }
                }
            });
        }
    });
    if let Some(message) = error.into_inner().map_err(|_| "worker panicked")? {
        return Err(message);
    }
    let all = results.into_inner().map_err(|_| "worker panicked")?;
    all.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|o| o.ok_or_else(|| "missing outcome".to_string()))
                .collect()
        })
        .collect()
}

/// Training rankings for every held-out prompt.
pub fn held_out_rankings(trace: &Trace) -> Vec<Vec<Vec<u16>>> {
    (0..trace.prompts.len())
        .map(|held| {
            let training: Vec<&PromptTrace> = trace
                .prompts
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != held)
                .map(|(_, p)| p)
                .collect();
            ranking(trace, &training)
        })
        .collect()
}

/// Uniform sizes: the percentages and the planning value, as `(label, slots)`.
pub fn sizes(experts: usize, planning: usize) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = PERCENTS
        .iter()
        .map(|&p| (format!("p{p}"), experts * p / 100))
        .collect();
    out.push(("plan".to_string(), planning));
    out
}

/// Greedy per-layer splits at the planning bytes for each held-out prompt: LRU from the
/// training stack distances of generated steps, static from the training ranking counts.
pub fn greedy_slots(trace: &Trace, planning: usize, policy: Kind) -> Vec<Vec<usize>> {
    let k = trace.top_k;
    let budget = trace.expert_bytes.iter().sum::<u64>() * planning as u64;
    (0..trace.prompts.len())
        .map(|held| {
            let training: Vec<&PromptTrace> = trace
                .prompts
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != held)
                .map(|(_, p)| p)
                .collect();
            let gains: Vec<Vec<u64>> = (0..trace.layers)
                .map(|layer| match policy {
                    Kind::Static => {
                        let mut counts = vec![0u64; trace.experts];
                        for p in &training {
                            for step in &p.steps {
                                for &e in trace.sel(step, layer) {
                                    counts[usize::from(e)] += 1;
                                }
                            }
                        }
                        counts.sort_unstable_by(|a, b| b.cmp(a));
                        // Slot s (s >= k) adds core expert s - k.
                        let mut gain = vec![0u64; k];
                        gain.extend(counts);
                        gain
                    }
                    _ => {
                        let mut hist = vec![0u64; trace.experts + 1];
                        for p in &training {
                            let d = stack_distances(trace, p, layer);
                            for (index, &distance) in d.iter().enumerate() {
                                if p.steps[index / k].decode && distance != COLD {
                                    hist[(distance as usize).min(trace.experts)] += 1;
                                }
                            }
                        }
                        hist
                    }
                })
                .collect();
            greedy_split(&trace.expert_bytes, budget, k, trace.experts, |layer, s| {
                gains[layer].get(s).copied().unwrap_or(0)
            })
        })
        .collect()
}

/// The jobs of the report: every policy, size, start, and admission rule without a page cache;
/// the page-cache sizes for empty starts with admission; and the greedy splits of LRU and static
/// at the planning size. A size whose tail cannot take the misses of one step is left out.
pub fn jobs(trace: &Trace, planning: usize) -> Vec<Job> {
    let (prompts, k) = (trace.prompts.len(), trace.top_k);
    let mut out = Vec::new();
    for policy in POLICIES {
        for (size, slots) in sizes(trace.experts, planning) {
            if slots < k || slots > trace.experts || slots - policy.core_size(slots, k) < k {
                continue;
            }
            for page_cache_gib in PAGE_CACHE_GIB {
                for warm in [false, true] {
                    for admit_prompt in [true, false] {
                        if page_cache_gib > 0 && (warm || !admit_prompt) {
                            continue;
                        }
                        // The page-cache baseline has no start state, prompt rule, or second level.
                        if policy.kind == Kind::PageOnly
                            && (page_cache_gib > 0 || warm || !admit_prompt)
                        {
                            continue;
                        }
                        out.push(Job {
                            policy,
                            split: "uniform",
                            size: size.clone(),
                            slots: vec![vec![slots; trace.layers]; prompts],
                            warm,
                            admit_prompt,
                            page_cache_gib,
                        });
                    }
                }
            }
        }
    }
    for (name, kind) in [("lru", Kind::Lru), ("static", Kind::Static)] {
        let Some(policy) = Policy::named(name) else {
            continue;
        };
        let slots = greedy_slots(trace, planning, kind);
        for warm in [false, true] {
            for admit_prompt in [true, false] {
                out.push(Job {
                    policy,
                    split: "greedy",
                    size: "plan".to_string(),
                    slots: slots.clone(),
                    warm,
                    admit_prompt,
                    page_cache_gib: 0,
                });
            }
        }
    }
    out
}

/// The text files of a report, by file name.
pub type Files = BTreeMap<&'static str, String>;

/// Runs the replay grid and returns the report files.
pub fn report(
    trace: &Trace,
    tokens: Option<&BTreeMap<u32, TokenLog>>,
    settings: &Settings,
    self_test: &[(&'static str, Result<(), String>)],
) -> Result<Files, String> {
    let k = trace.top_k;
    let step_accesses = (trace.layers * k) as u64;
    let rankings = held_out_rankings(trace);
    let jobs = jobs(trace, settings.planning_slots);
    let outcomes = run_jobs(trace, &rankings, &jobs, settings.threads)?;
    let ids: Vec<u32> = trace.prompts.iter().map(|p| p.id).collect();
    let repeated: Vec<Option<f64>> = ids
        .iter()
        .map(|id| {
            tokens
                .and_then(|t| t.get(id))
                .map(|log| repetition(&log.outputs))
        })
        .collect();
    let decision: Vec<bool> = repeated
        .iter()
        .map(|r| r.is_none_or(|r| r <= settings.repetition_limit))
        .collect();

    let mut results = String::from(
        "policy\tsplit\tsize\tslots\tstart\tadmission\tpage_cache_gib\tscope\tdecode_tokens\t\
         decode_hit_rate\tprompt_hit_rate\tfirst16\tfirst64\tfirst256\tmisses_mean\tmisses_p50\t\
         misses_p95\tmisses_max\tmiss_bytes_mean\tpage_cache_bytes_mean\tssd_bytes_mean",
    );
    for bw in &settings.bandwidths {
        let _ = write!(results, "\tread_s_{0}\ttok_s_{0}", bw.name);
    }
    results.push('\n');
    let mut layers_tsv = String::from("policy\tsplit\tsize\tstart\tadmission");
    for layer in 0..trace.layers {
        let _ = write!(layers_tsv, "\tl{layer}");
    }
    layers_tsv.push('\n');
    // Decision rows: (job index, summary) at the planning size without a page cache.
    let mut decision_rows: Vec<(usize, Summary)> = Vec::new();
    let mut all_rows: Vec<(usize, Summary)> = Vec::new();
    for (j, job) in jobs.iter().enumerate() {
        let slots_label = if job.split == "uniform" {
            job.slots[0][0].to_string()
        } else {
            "split".to_string()
        };
        let start = if job.warm { "warm" } else { "empty" };
        let admission = if job.admit_prompt { "admit" } else { "bypass" };
        let mut scopes: Vec<(String, Summary)> = Vec::new();
        let mut all = Summary::default();
        let mut chosen = Summary::default();
        for (p, outcome) in outcomes[j].iter().enumerate() {
            let mut one = Summary::default();
            one.add(outcome, step_accesses);
            scopes.push((ids[p].to_string(), one));
            all.add(outcome, step_accesses);
            if decision[p] {
                chosen.add(outcome, step_accesses);
            }
        }
        if job.page_cache_gib == 0 {
            let _ = write!(
                layers_tsv,
                "{}\t{}\t{}\t{start}\t{admission}",
                job.policy.name, job.split, job.size
            );
            for (h, a) in chosen.layer_hits.iter().zip(&chosen.layer_accesses) {
                let _ = write!(layers_tsv, "\t{:.4}", rate(*h, *a));
            }
            layers_tsv.push('\n');
            all_rows.push((j, all.clone()));
            if job.size == "plan" {
                decision_rows.push((j, chosen.clone()));
            }
        }
        scopes.push(("all".to_string(), all));
        scopes.push(("decision".to_string(), chosen));
        for (scope, s) in &scopes {
            let (page_cache, ssd) = s.bytes_per_token();
            let mean =
                s.misses.iter().map(|&m| f64::from(m)).sum::<f64>() / s.misses.len().max(1) as f64;
            let _ = write!(
                results,
                "{}\t{}\t{}\t{slots_label}\t{start}\t{admission}\t{}\t{scope}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{mean:.3}\t{}\t{}\t{}\t{:.0}\t{page_cache:.0}\t{ssd:.0}",
                job.policy.name,
                job.split,
                job.size,
                job.page_cache_gib,
                s.decode_tokens,
                s.hit_rate(),
                rate(s.prompt_hits, s.prompt_accesses),
                rate(s.windows[0].0, s.windows[0].1),
                rate(s.windows[1].0, s.windows[1].1),
                rate(s.windows[2].0, s.windows[2].1),
                percentile(&s.misses, 0.5),
                percentile(&s.misses, 0.95),
                s.misses.iter().copied().max().unwrap_or(0),
                page_cache + ssd,
            );
            for bw in &settings.bandwidths {
                let read = read_seconds(s, bw.bytes_per_second, settings.page_cache_bandwidth);
                let _ = write!(
                    results,
                    "\t{read:.6}\t{:.4}",
                    1.0 / (settings.t_fixed + read)
                );
            }
            results.push('\n');
        }
    }

    let mut files = Files::new();
    files.insert("results.tsv", results);
    files.insert("layers.tsv", layers_tsv);
    files.insert("reuse.tsv", reuse(trace));
    files.insert("distinct.tsv", distinct(trace));
    let checks = checks(
        trace, tokens, &repeated, settings, self_test, &jobs, &all_rows,
    );
    files.insert("checks.txt", checks);
    files.insert(
        "summary.txt",
        summary(trace, settings, &jobs, &decision_rows, &decision, &ids),
    );
    Ok(files)
}

/// The LRU stack-distance histogram of generated-step accesses over all prompts and layers.
fn reuse(trace: &Trace) -> String {
    let k = trace.top_k;
    let mut hist = vec![0u64; trace.experts];
    let mut cold = 0u64;
    for prompt in &trace.prompts {
        for layer in 0..trace.layers {
            for (index, &d) in stack_distances(trace, prompt, layer).iter().enumerate() {
                if !prompt.steps[index / k].decode {
                    continue;
                }
                if d == COLD {
                    cold += 1;
                } else {
                    hist[(d as usize).min(trace.experts - 1)] += 1;
                }
            }
        }
    }
    let mut out = String::from("distance\tcount\n");
    for (d, count) in hist.iter().enumerate() {
        let _ = writeln!(out, "{d}\t{count}");
    }
    let _ = writeln!(out, "cold\t{cold}");
    out
}

/// Distinct experts per layer and prompt, over all steps and over generated steps.
fn distinct(trace: &Trace) -> String {
    let mut out = String::from("prompt\tlayer\tdistinct_all\tdistinct_decode\n");
    for prompt in &trace.prompts {
        for layer in 0..trace.layers {
            let mut all = vec![false; trace.experts];
            let mut decode = vec![false; trace.experts];
            for step in &prompt.steps {
                for &e in trace.sel(step, layer) {
                    all[usize::from(e)] = true;
                    if step.decode {
                        decode[usize::from(e)] = true;
                    }
                }
            }
            let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
            let _ = writeln!(
                out,
                "{}\t{layer}\t{}\t{}",
                prompt.id,
                count(&all),
                count(&decode)
            );
        }
    }
    out
}

/// The T0-P1 to T0-P3 checks that the trace and the replay can show.
fn checks(
    trace: &Trace,
    tokens: Option<&BTreeMap<u32, TokenLog>>,
    repeated: &[Option<f64>],
    settings: &Settings,
    self_test: &[(&'static str, Result<(), String>)],
    jobs: &[Job],
    all_rows: &[(usize, Summary)],
) -> String {
    let mut out = String::new();
    let steps: usize = trace.prompts.iter().map(|p| p.steps.len()).sum();
    let generated: usize = trace
        .prompts
        .iter()
        .map(|p| p.steps.iter().filter(|s| s.decode).count())
        .sum();
    let longest = trace
        .prompts
        .iter()
        .map(|p| p.steps.iter().filter(|s| !s.decode).count() + 1)
        .max()
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "t0_p1_rows=pass rows={} steps={steps} layers={} (complete steps, distinct ids, finite values, gate sums)",
        steps * trace.layers,
        trace.layers
    );
    let p2 = generated >= 2048 && trace.prompts.len() >= 5 && longest >= 2048;
    let _ = writeln!(
        out,
        "t0_p2_coverage={} generated={generated} prompts={} longest_prompt_tokens={longest}",
        if p2 { "pass" } else { "fail" },
        trace.prompts.len()
    );
    for (prompt, r) in trace.prompts.iter().zip(repeated) {
        let tokens_note = tokens
            .and_then(|t| t.get(&prompt.id))
            .map_or("no tokens.tsv row".to_string(), |log| {
                format!("prompt_tokens={}", log.prompt_tokens)
            });
        let _ = writeln!(
            out,
            "repetition_prompt{}={} {}{tokens_note}",
            prompt.id,
            r.map_or("unavailable".to_string(), |r| format!("{r:.4}")),
            if r.is_some_and(|r| r > settings.repetition_limit) {
                "excluded "
            } else {
                ""
            },
        );
    }
    let failed: Vec<&str> = self_test
        .iter()
        .filter(|(_, r)| r.is_err())
        .map(|(name, _)| *name)
        .collect();
    let _ = writeln!(
        out,
        "t0_p3_self_test={} cases={} failed={failed:?}",
        if failed.is_empty() { "pass" } else { "fail" },
        self_test.len()
    );
    // Belady bounds every policy that starts from the same contents: the same start, and for an
    // empty start no static core.
    let mut violations = Vec::new();
    for (j, s) in all_rows {
        let job = &jobs[*j];
        if matches!(job.policy.kind, Kind::Belady | Kind::PoolBelady) || job.split != "uniform" {
            continue;
        }
        if !job.warm && job.policy.kind == Kind::Static {
            continue;
        }
        let bound = all_rows.iter().find(|(b, _)| {
            let other = &jobs[*b];
            other.policy.kind == job.policy.kind.bound()
                && other.size == job.size
                && other.warm == job.warm
                && other.admit_prompt == job.admit_prompt
        });
        // Belady minimizes the misses of every admitted step, so the bound is on all hits.
        if let Some((_, b)) = bound
            && s.decode_hits + s.prompt_hits > b.decode_hits + b.prompt_hits
        {
            violations.push(format!(
                "{} {} warm={} admit={}",
                job.policy.name, job.size, job.warm, job.admit_prompt
            ));
        }
    }
    let _ = writeln!(
        out,
        "t0_p3_belady_bound={} violations={violations:?}",
        if violations.is_empty() {
            "pass"
        } else {
            "fail"
        }
    );
    // The smallest size, the planning size, and the largest size of the direct LRU replay.
    let direct: Vec<&(usize, Summary)> = all_rows
        .iter()
        .filter(|(j, _)| {
            let job = &jobs[*j];
            job.policy.kind == Kind::Lru && job.split == "uniform" && !job.warm && job.admit_prompt
        })
        .collect();
    let slots_of = |row: &&(usize, Summary)| jobs[row.0].slots[0][0];
    let chosen = [
        direct.iter().copied().min_by_key(slots_of),
        direct
            .iter()
            .copied()
            .find(|row| jobs[row.0].size == "plan"),
        direct.iter().copied().max_by_key(slots_of),
    ];
    let mut curve = Vec::new();
    for row in chosen.into_iter().flatten() {
        let (j, s) = row;
        let slots = jobs[*j].slots[0][0];
        let from_distances = lru_hits_from_distances(trace, slots);
        curve.push((
            jobs[*j].size.clone(),
            slots,
            s.decode_hits + s.prompt_hits,
            from_distances,
        ));
    }
    let equal = curve.len() == 3 && curve.iter().all(|(_, _, a, b)| a == b);
    let _ = writeln!(
        out,
        "t0_p3_lru_curve={} sizes={curve:?}",
        if equal { "pass" } else { "fail" }
    );
    out
}

/// LRU hits of all steps of all prompts at `slots` slots from the stack distances.
pub fn lru_hits_from_distances(trace: &Trace, slots: usize) -> u64 {
    let mut hits = 0;
    for prompt in &trace.prompts {
        for layer in 0..trace.layers {
            hits += stack_distances(trace, prompt, layer)
                .iter()
                .filter(|&&d| d != COLD && (d as usize) < slots)
                .count() as u64;
        }
    }
    hits
}

/// The decision value and the gates.
fn summary(
    trace: &Trace,
    settings: &Settings,
    jobs: &[Job],
    decision_rows: &[(usize, Summary)],
    decision: &[bool],
    ids: &[u32],
) -> String {
    let mut out = String::new();
    let share = settings.planning_slots as f64 / trace.experts as f64;
    let miss_bytes = trace.step_bytes() as f64;
    let included: Vec<u32> = ids
        .iter()
        .zip(decision)
        .filter(|&(_, &d)| d)
        .map(|(id, _)| *id)
        .collect();
    let _ = writeln!(
        out,
        "decision_prompts={included:?}\nplanning_slots={}\ncapacity_share={share:.4}\n\
         miss_bytes_per_token={miss_bytes:.0}\nt_fixed={}",
        settings.planning_slots, settings.t_fixed
    );
    for warm in [false, true] {
        let best = decision_rows
            .iter()
            .filter(|(j, _)| jobs[*j].policy.kind.is_candidate() && jobs[*j].warm == warm)
            .max_by(|a, b| a.1.hit_rate().total_cmp(&b.1.hit_rate()));
        let start = if warm { "warm" } else { "empty" };
        for (j, s) in decision_rows
            .iter()
            .filter(|(j, _)| jobs[*j].warm == warm && jobs[*j].split == "uniform")
        {
            let job = &jobs[*j];
            let _ = writeln!(
                out,
                "plan_hit_{start}_{}_{}={:.6}",
                job.policy.name,
                if job.admit_prompt { "admit" } else { "bypass" },
                s.hit_rate()
            );
        }
        let Some((j, s)) = best else {
            continue;
        };
        let job = &jobs[*j];
        let h = s.hit_rate();
        let _ = writeln!(
            out,
            "h_star_{start}={h:.6} policy={} split={} admission={}",
            job.policy.name,
            job.split,
            if job.admit_prompt { "admit" } else { "bypass" }
        );
        for name in ["lru", "static"] {
            let best_of = |split: &str| {
                decision_rows
                    .iter()
                    .filter(|(j, _)| {
                        jobs[*j].policy.name == name
                            && jobs[*j].split == split
                            && jobs[*j].warm == warm
                    })
                    .map(|(_, s)| s.hit_rate())
                    .fold(f64::NAN, f64::max)
            };
            let _ = writeln!(
                out,
                "split_gain_{name}_{start}={:.6} (greedy minus uniform; Q2 needs at least 0.02)",
                best_of("greedy") - best_of("uniform")
            );
        }
        let cold = settings
            .cold
            .as_ref()
            .and_then(|name| settings.bandwidths.iter().find(|b| &b.name == name));
        match cold {
            Some(bw) => {
                let g = gates(h, settings.t_fixed, miss_bytes, bw.bytes_per_second, share);
                let _ = writeln!(
                    out,
                    "gates_{start} bw_cold={}={} g1_seconds={:.6} t0_g1={} g2_saved={:.6} t0_g2={}",
                    bw.name,
                    bw.bytes_per_second,
                    g.g1_seconds,
                    if g.g1 { "go" } else { "no" },
                    g.g2_saved,
                    if g.g2 { "go" } else { "no" }
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "gates_{start}=not evaluated: name the T0b best-depth bandwidth with --bw-cold"
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selftest::seeded;

    #[test]
    fn the_page_cache_baseline_runs_only_from_an_empty_start_with_admission() {
        let page_only: Vec<Job> = jobs(&seeded(), 6)
            .into_iter()
            .filter(|job| job.policy.kind == Kind::PageOnly)
            .collect();
        assert!(!page_only.is_empty(), "no page-only job");
        for job in &page_only {
            assert!(
                !job.warm && job.admit_prompt && job.page_cache_gib == 0,
                "{job:?}"
            );
        }
    }

    #[test]
    fn the_report_of_the_seeded_trace_passes_its_checks() {
        let trace = seeded();
        let settings = Settings {
            planning_slots: 6,
            t_fixed: 0.149,
            bandwidths: vec![Bandwidth {
                name: "ssd".to_string(),
                bytes_per_second: 1e9,
            }],
            page_cache_bandwidth: 7e9,
            cold: Some("ssd".to_string()),
            repetition_limit: 0.5,
            threads: 2,
        };
        let files = report(&trace, None, &settings, &[("case", Ok(()))]).unwrap();
        let checks = &files["checks.txt"];
        // 2 prompts of 40 prompt steps and 60 generated steps: 41 prompt tokens each.
        for line in [
            "t0_p1_rows=pass rows=400 steps=200 layers=2",
            "t0_p2_coverage=fail generated=120 prompts=2 longest_prompt_tokens=41",
            "t0_p3_self_test=pass cases=1",
            "t0_p3_belady_bound=pass",
            "t0_p3_lru_curve=pass",
        ] {
            assert!(checks.contains(line), "{line} missing from {checks}");
        }
        let summary = &files["summary.txt"];
        assert!(
            summary.contains("h_star_empty=")
                && summary.contains("gates_empty bw_cold=ssd=1000000000"),
            "{summary}"
        );
        // Every policy has a hit rate at the planning size, in the unit of the decision value.
        for line in [
            "plan_hit_empty_lru_admit=",
            "plan_hit_empty_pool_admit=",
            "plan_hit_warm_belady-pool_bypass=",
            "plan_hit_empty_page-only_admit=",
            "plan_hit_warm_hot-h64_bypass=",
        ] {
            assert!(summary.contains(line), "{line} missing from {summary}");
        }
        let rows = files["results.tsv"].lines().count();
        assert_eq!(
            rows,
            1 + jobs(&trace, 6).len() * 4,
            "one row per job and scope"
        );
    }

    use crate::selftest::{StepList, synthetic};
    use crate::sim::DecodeStep;

    fn job(policy: &str, split: &'static str, warm: bool, admit_prompt: bool) -> Job {
        Job {
            policy: Policy::named(policy).unwrap(),
            split,
            size: "plan".to_string(),
            slots: Vec::new(),
            warm,
            admit_prompt,
            page_cache_gib: 0,
        }
    }

    /// A summary with `permille / 1000` as its generated hit rate.
    fn with_rate(permille: u64) -> Summary {
        Summary {
            decode_hits: permille,
            decode_accesses: 1000,
            ..Summary::default()
        }
    }

    #[test]
    fn sizes_are_floors_of_the_expert_share_and_end_with_the_planning_value() {
        // 512 experts: 512 * p / 100 is 51.2, 76.8, 102.4, 128, 153.6, 179.2, 204.8, 230.4, 256
        // for p = 10, 15, ..., 50. The floors are 51, 76, 102, 128, 153, 179, 204, 230, 256. A
        // divisor of 99 would give 77 for 15 percent.
        let got: Vec<usize> = sizes(512, 193).iter().map(|(_, s)| *s).collect();
        assert_eq!(got, vec![51, 76, 102, 128, 153, 179, 204, 230, 256, 193]);
        let labels: Vec<String> = sizes(512, 193).into_iter().map(|(l, _)| l).collect();
        assert_eq!(labels[0], "p10");
        assert_eq!(labels[9], "plan");
    }

    #[test]
    fn read_seconds_adds_the_page_cache_and_ssd_terms() {
        // 2 generated tokens with 600 page-cache bytes and 400 SSD bytes: 300 and 200 bytes per
        // token. At 50 B/s for the page cache and 100 B/s for the SSD: 300 / 50 + 200 / 100 =
        // 6 + 2 = 8 seconds.
        let summary = Summary {
            decode_tokens: 2,
            page_cache_bytes: 600,
            ssd_bytes: 400,
            ..Summary::default()
        };
        assert_eq!(summary.bytes_per_token(), (300.0, 200.0));
        assert_eq!(read_seconds(&summary, 100.0, 50.0), 8.0);
        // No generated token: the divisor is 1, so the totals are the per-token bytes.
        assert_eq!(Summary::default().bytes_per_token(), (0.0, 0.0));
    }

    #[test]
    fn windows_hold_the_first_16_64_and_256_generated_steps() {
        // 300 generated steps with 1 hit each and 2 accesses per step. The windows hold steps
        // 0 to 15, 0 to 63, and 0 to 255: 16, 64, and 256 steps, so (16, 32), (64, 128), and
        // (256, 512) hits and accesses. All 300 steps count in the totals.
        let step = DecodeStep {
            hits: 1,
            misses: 1,
            page_cache_bytes: 0,
            ssd_bytes: 0,
        };
        let outcome = Outcome {
            decode: vec![step; 300],
            ..Outcome::default()
        };
        let mut summary = Summary::default();
        summary.add(&outcome, 2);
        assert_eq!(summary.windows, [(16, 32), (64, 128), (256, 512)]);
        assert_eq!((summary.decode_hits, summary.decode_accesses), (300, 600));
        assert_eq!(summary.decode_tokens, 300);
        assert_eq!(summary.misses.len(), 300);
    }

    #[test]
    fn layer_sums_add_the_outcomes_of_each_prompt() {
        // Prompt A hits 1 and 2 of 4 accesses in layers 0 and 1; prompt B hits 3 and 0 of 4.
        // The sums are hits 4 and 2 of 8 and 8 accesses; prompt hits add up as 5 + 7 of 6 + 6.
        let outcome = |hits: [u64; 2], prompt_hits: u64| Outcome {
            layer_hits: hits.to_vec(),
            layer_accesses: vec![4, 4],
            prompt_hits,
            prompt_accesses: 6,
            ..Outcome::default()
        };
        let mut summary = Summary::default();
        summary.add(&outcome([1, 2], 5), 2);
        summary.add(&outcome([3, 0], 7), 2);
        assert_eq!(summary.layer_hits, vec![4, 2]);
        assert_eq!(summary.layer_accesses, vec![8, 8]);
        assert_eq!((summary.prompt_hits, summary.prompt_accesses), (12, 12));
    }

    /// Settings of the `summary` test: 2 of 10 experts, so the capacity share is 0.2.
    fn summary_settings() -> Settings {
        Settings {
            planning_slots: 2,
            t_fixed: 0.15,
            bandwidths: vec![Bandwidth {
                name: "cold".to_string(),
                bytes_per_second: 4000.0,
            }],
            page_cache_bandwidth: 1.0,
            cold: Some("cold".to_string()),
            repetition_limit: 0.5,
            threads: 1,
        }
    }

    #[test]
    fn the_summary_takes_the_best_policy_of_each_start_and_the_split_gains() {
        // Two layers with 100 and 300 bytes per expert and 2 experts per step: a step reads
        // 2 * (100 + 300) = 800 bytes.
        let mut trace = synthetic(2, 10, 2, Vec::new());
        trace.expert_bytes = vec![100, 300];
        // (policy, split, warm, admit, decode hit rate in permille). Belady is the bound and
        // never the decision value, even at 900.
        let table: [(&str, &'static str, bool, bool, u64); 11] = [
            ("lru", "uniform", false, true, 300),
            ("lfu", "uniform", false, false, 500),
            ("belady", "uniform", false, true, 900),
            ("static", "greedy", true, true, 800),
            ("lru", "uniform", true, true, 400),
            ("lru", "greedy", false, true, 340),
            ("lru", "greedy", true, true, 450),
            ("static", "uniform", false, true, 200),
            ("static", "greedy", false, true, 450),
            ("static", "uniform", true, true, 550),
            ("lru", "uniform", false, false, 250),
        ];
        let jobs: Vec<Job> = table
            .iter()
            .map(|&(p, split, warm, admit, _)| job(p, split, warm, admit))
            .collect();
        let rows: Vec<(usize, Summary)> = table
            .iter()
            .enumerate()
            .map(|(j, row)| (j, with_rate(row.4)))
            .collect();
        let text = summary(
            &trace,
            &summary_settings(),
            &jobs,
            &rows,
            &[true, false, true],
            &[7, 8, 9],
        );
        // Prompt 8 is excluded. The share is 2 / 10 = 0.2.
        // Empty start: the best non-Belady rate is 0.500 (lfu uniform, bypass). Gates at
        // h = 0.5: g1 = 0.15 + 0.5 * 800 / 4000 = 0.25 (above 0.200), and g2 saves
        // 0.5 * 800 / 4000 = 0.1 s with 0.5 >= 1.5 * 0.2.
        // Split gains of the empty start: lru greedy 0.34 minus the best uniform 0.30 (the
        // bypass row at 0.25 is lower) = 0.04; static greedy 0.45 minus uniform 0.20 = 0.25.
        // Warm start: the best rate is 0.800 (static greedy, admit). Gates at h = 0.8:
        // g1 = 0.15 + 0.2 * 800 / 4000 = 0.19 (go), and g2 saves 0.8 * 800 / 4000 = 0.16 s.
        // Split gains of the warm start: lru 0.45 - 0.40 = 0.05; static 0.80 - 0.55 = 0.25.
        // Before each `h_star` line come the `plan_hit` lines: one per uniform row of the start,
        // in row order, with the rate of the row. Empty start: lru admit 0.30, lfu bypass 0.50,
        // belady admit 0.90, static admit 0.20, lru bypass 0.25. Warm start: lru admit 0.40,
        // static admit 0.55. The greedy rows have no line.
        let want = "decision_prompts=[7, 9]\n\
                    planning_slots=2\n\
                    capacity_share=0.2000\n\
                    miss_bytes_per_token=800\n\
                    t_fixed=0.15\n\
                    plan_hit_empty_lru_admit=0.300000\n\
                    plan_hit_empty_lfu_bypass=0.500000\n\
                    plan_hit_empty_belady_admit=0.900000\n\
                    plan_hit_empty_static_admit=0.200000\n\
                    plan_hit_empty_lru_bypass=0.250000\n\
                    h_star_empty=0.500000 policy=lfu split=uniform admission=bypass\n\
                    split_gain_lru_empty=0.040000 (greedy minus uniform; Q2 needs at least 0.02)\n\
                    split_gain_static_empty=0.250000 (greedy minus uniform; Q2 needs at least 0.02)\n\
                    gates_empty bw_cold=cold=4000 g1_seconds=0.250000 t0_g1=no g2_saved=0.100000 t0_g2=go\n\
                    plan_hit_warm_lru_admit=0.400000\n\
                    plan_hit_warm_static_admit=0.550000\n\
                    h_star_warm=0.800000 policy=static split=greedy admission=admit\n\
                    split_gain_lru_warm=0.050000 (greedy minus uniform; Q2 needs at least 0.02)\n\
                    split_gain_static_warm=0.250000 (greedy minus uniform; Q2 needs at least 0.02)\n\
                    gates_warm bw_cold=cold=4000 g1_seconds=0.190000 t0_g1=go g2_saved=0.160000 t0_g2=go\n";
        assert_eq!(text, want);
    }

    #[test]
    fn the_summary_does_not_evaluate_gates_without_a_named_cold_bandwidth() {
        let mut trace = synthetic(1, 10, 1, Vec::new());
        trace.expert_bytes = vec![100];
        let jobs = vec![job("lru", "uniform", false, true)];
        let mut settings = summary_settings();
        settings.cold = None;
        let text = summary(
            &trace,
            &settings,
            &jobs,
            &[(0, with_rate(300))],
            &[true],
            &[1],
        );
        assert!(
            text.contains(
                "gates_empty=not evaluated: name the T0b best-depth bandwidth with --bw-cold\n"
            ) && !text.contains("h_star_warm"),
            "{text}"
        );
    }

    #[test]
    fn prompts_above_the_repetition_limit_leave_the_decision_value() {
        // Three prompts of 1 prompt step and 2, 3, and 4 generated steps. Prompt 1 has no token
        // log and stays in. Prompt 2 generates 1, 1, 1, 1, 1, 2, 3, 4: its 5 four-grams are
        // 1111, 1111, 1112, 1123, 1234 and one repeats, so the share is 1 / 5 = 0.2, which
        // equals the limit and stays in. Prompt 3 generates twelve 7s: 9 four-grams, 8 repeat,
        // 8 / 9 = 0.8889 above the limit, so it leaves.
        let steps = |decode: usize| {
            let mut out = vec![(false, vec![1, 2])];
            out.extend((0..decode).map(|i| (true, vec![i as u16, 3])));
            out
        };
        let mut trace = synthetic(2, 10, 1, vec![(1, steps(2)), (2, steps(3)), (3, steps(4))]);
        trace.expert_bytes = vec![100, 300];
        let log = |outputs: Vec<u32>| TokenLog {
            prompt_tokens: 5,
            outputs,
        };
        let tokens = BTreeMap::from([
            (2, log(vec![1, 1, 1, 1, 1, 2, 3, 4])),
            (3, log(vec![7; 12])),
        ]);
        let settings = Settings {
            planning_slots: 3,
            repetition_limit: 0.2,
            ..summary_settings()
        };
        let files = report(&trace, Some(&tokens), &settings, &[]).unwrap();
        let checks = &files["checks.txt"];
        for line in [
            "repetition_prompt1=unavailable no tokens.tsv row\n",
            "repetition_prompt2=0.2000 prompt_tokens=5\n",
            "repetition_prompt3=0.8889 excluded prompt_tokens=5\n",
        ] {
            assert!(checks.contains(line), "{line} missing from {checks}");
        }
        assert!(files["summary.txt"].starts_with("decision_prompts=[1, 2]\n"));
        // The decision scope of a row counts the generated steps of prompts 1 and 2: 2 + 3 = 5.
        // The all scope counts 2 + 3 + 4 = 9.
        let decode_tokens = |scope: &str| -> String {
            let line = files["results.tsv"]
                .lines()
                .find(|l| {
                    l.starts_with("lru\tuniform\tplan\t3\tempty\tadmit\t0\t")
                        && l.split('\t').nth(7) == Some(scope)
                })
                .unwrap();
            line.split('\t').nth(8).unwrap().to_string()
        };
        assert_eq!(decode_tokens("decision"), "5");
        assert_eq!(decode_tokens("all"), "9");
    }

    /// Two layers with 10 and 20 bytes per expert, four experts, one expert per step. A step
    /// lists the expert of layer 0 and the expert of layer 1; the first steps are prompt steps.
    fn greedy_trace(prompts: Vec<StepList>) -> Trace {
        let mut trace = synthetic(
            2,
            4,
            1,
            prompts
                .into_iter()
                .enumerate()
                .map(|(i, p)| (i as u32, p))
                .collect(),
        );
        trace.expert_bytes = vec![10, 20];
        trace
    }

    #[test]
    fn the_lru_split_uses_the_generated_stack_distances_of_the_other_prompts() {
        // Planning 2 slots: budget 2 * (10 + 20) = 60, the minimum of 1 slot per layer costs 30,
        // 30 bytes are left. The gain of slot s+1 is the count of generated accesses of the
        // training prompts with stack distance s (cold accesses and prompt steps do not count).
        // Prompt 0: layer 0 reads 3,3,2 | 0,3 and layer 1 reads 3,0,0 | 3,1 (| ends the prompt
        // steps). Prompt 1: layer 0 reads 1,2,0 | 1,1 and layer 1 reads 3,0,3 | 3,2.
        let trace = greedy_trace(vec![
            vec![
                (false, vec![3, 3]),
                (false, vec![3, 0]),
                (false, vec![2, 0]),
                (true, vec![0, 3]),
                (true, vec![3, 1]),
            ],
            vec![
                (false, vec![1, 3]),
                (false, vec![2, 0]),
                (false, vec![0, 3]),
                (true, vec![1, 3]),
                (true, vec![1, 2]),
            ],
        ]);
        // Held out prompt 0 trains on prompt 1. Layer 0 generated distances: 1 comes after 2, 0
        // so distance 2, then 1 again at distance 0: histogram [1, 0, 1]. Layer 1: 3 at distance
        // 0, then 2 is cold: [1]. With gain(l, s) = hist[s], round 1 (left 30): layer 0 has
        // hist[1] = 0 and layer 1 has hist[1] = 0, a tie that goes to layer 0 (slots 2,1; left
        // 20). Round 2: layer 0 has hist[2] / 10 = 0.1 against layer 1 with 0 (slots 3,1; left
        // 10). Round 3: layer 0 has hist[3] = 0 and layer 1 does not fit 20 bytes in 10: slots
        // 4,1. Layer 0 is at the maximum of 4.
        // Held out prompt 1 trains on prompt 0. Layer 0: 3 repeats at distance 2 (after 2, 0):
        // [0, 0, 1]. Layer 1: 3 returns at distance 1 (after 0): [0, 1]. Round 1: layer 0 has
        // hist[1] = 0, layer 1 has hist[1] / 20 = 0.05, so layer 1 wins (slots 1,2; left 10).
        // Round 2: layer 0 fits 10 bytes with gain 0; layer 1 does not fit: slots 2,2.
        assert_eq!(
            greedy_slots(&trace, 2, Kind::Lru),
            vec![vec![4, 1], vec![2, 2]]
        );
    }

    #[test]
    fn the_static_split_uses_the_selection_counts_of_the_other_prompts() {
        // Planning 3 slots: budget 3 * 30 = 90, the minimum costs 30, 60 are left. With top_k 1
        // the gain of slot s+1 is the s-th largest selection count of the training prompts, over
        // prompt and generated steps (slot 2 adds core expert 0, whose count is the largest).
        // Prompt 0 reads layer 0: 3,1,1,3 and layer 1: 3,2,1,2. Prompt 1 reads layer 0: 3,1,0,3
        // and layer 1: 0,2,2,1.
        let trace = greedy_trace(vec![
            vec![
                (false, vec![3, 3]),
                (false, vec![1, 2]),
                (true, vec![1, 1]),
                (true, vec![3, 2]),
            ],
            vec![
                (false, vec![3, 0]),
                (false, vec![1, 2]),
                (true, vec![0, 2]),
                (true, vec![3, 1]),
            ],
        ]);
        // Held out prompt 0 trains on prompt 1: layer 0 counts sorted [2, 1, 1, 0] and layer 1
        // counts [2, 1, 1, 0]; gain(l, s) = counts[s - 1]. Round 1 (left 60): 2/10 = 0.2 against
        // 2/20 = 0.1, layer 0 (2,1; left 50). Round 2: 1/10 = 0.1 against 2/20 = 0.1, tie to
        // layer 0 (3,1; left 40). Round 3: 1/10 against 0.1, tie to layer 0 (4,1; left 30).
        // Round 4: layer 0 is at the maximum, layer 1 takes 2/20 (4,2; left 10). Round 5: layer 1
        // needs 20 bytes.
        // Held out prompt 1 trains on prompt 0: layer 0 counts [2, 2, 0, 0] and layer 1 counts
        // [2, 1, 1, 0]. Round 1: 2/10 against 2/20, layer 0 (2,1; left 50). Round 2: 2/10
        // against 0.1, layer 0 (3,1; left 40). Round 3: layer 0 has 0/10 and layer 1 has 2/20,
        // layer 1 (3,2; left 20). Round 4: layer 0 has 0 and layer 1 has 1/20, layer 1 (3,3;
        // left 0).
        assert_eq!(
            greedy_slots(&trace, 3, Kind::Static),
            vec![vec![4, 2], vec![3, 3]]
        );
    }
}
