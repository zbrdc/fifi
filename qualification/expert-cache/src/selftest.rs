//! Self-test of the replay on synthetic traces with hand-derived values (T0-P3). `expert-cache
//! self-test` runs every case, and `cargo test` runs each case as one test.

use crate::cache::{Core, Kind, Policy};
use crate::report::{gates, held_out_rankings, lru_hits_from_distances, percentile};
use crate::sim::{Outcome, SimConfig, simulate};
use crate::trace::{PromptTrace, Step, Trace, repetition};

/// Steps of a synthetic prompt: `(decode, ids)`.
pub type StepList = Vec<(bool, Vec<u16>)>;

/// One self-test case.
pub type Case = fn() -> Result<(), String>;

/// A one-layer trace with one expert per step unless `top_k` says otherwise. Each prompt is a
/// list of `(decode, ids)` steps.
pub fn synthetic(
    layers: usize,
    experts: usize,
    top_k: usize,
    prompts: Vec<(u32, StepList)>,
) -> Trace {
    Trace {
        header: Vec::new(),
        layers,
        experts,
        top_k,
        expert_bytes: vec![100; layers],
        prompts: prompts
            .into_iter()
            .map(|(id, steps)| PromptTrace {
                id,
                steps: steps
                    .into_iter()
                    .map(|(decode, sel)| Step { decode, sel })
                    .collect(),
            })
            .collect(),
    }
}

/// A one-layer, one-expert-per-step trace of generated steps.
fn decode_only(experts: usize, ids: &[u16]) -> Trace {
    let steps = ids.iter().map(|&e| (true, vec![e])).collect();
    synthetic(1, experts, 1, vec![(0, steps)])
}

/// Replays prompt `prompt` of `trace` with `slots` slots per layer, an id-order ranking, an empty
/// start, admission, and no page cache.
fn run(trace: &Trace, policy: Policy, slots: usize, prompt: usize) -> Result<Outcome, String> {
    run_slots(trace, policy, &vec![slots; trace.layers], prompt, false)
}

/// Like `run`, with the slots of each layer and the start state given.
fn run_slots(
    trace: &Trace,
    policy: Policy,
    slots: &[usize],
    prompt: usize,
    warm: bool,
) -> Result<Outcome, String> {
    let ranked: Vec<Vec<u16>> = vec![(0..trace.experts as u16).collect(); trace.layers];
    let config = SimConfig {
        policy,
        slots,
        warm,
        admit_prompt: true,
        page_cache_bytes: 0,
    };
    simulate(trace, &trace.prompts[prompt], &ranked, &config)
}

fn policy(name: &str) -> Result<Policy, String> {
    Policy::named(name).ok_or_else(|| format!("no policy {name}"))
}

/// All hits of an outcome: generated and prompt steps.
fn hits(outcome: &Outcome) -> u64 {
    outcome
        .decode
        .iter()
        .map(|s| u64::from(s.hits))
        .sum::<u64>()
        + outcome.prompt_hits
}

fn expect(what: &str, got: u64, want: u64) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{what}: {got} hits, expected {want}"))
    }
}

/// The cycle 0, 1, 2, 3 three times. LRU with 3 slots evicts each expert just before its next use
/// and scores 0; with 4 slots only the first 4 accesses miss.
pub fn lru_cycle() -> Result<(), String> {
    let trace = decode_only(4, &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    expect("lru 3 slots", hits(&run(&trace, policy("lru")?, 3, 0)?), 0)?;
    expect("lru 4 slots", hits(&run(&trace, policy("lru")?, 4, 0)?), 8)
}

/// The same cycle under Belady with 3 slots: steps 3, 6, and 9 miss after the 3 cold misses; each
/// evicts the resident used furthest ahead (2, then 1, then 0), so steps 4, 5, 7, 8, 10, and 11
/// hit: 6 hits.
pub fn belady_cycle() -> Result<(), String> {
    let trace = decode_only(4, &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    expect(
        "belady 3 slots",
        hits(&run(&trace, policy("belady")?, 3, 0)?),
        6,
    )
}

/// Two experts alternate 10 times: 1 slot never hits, 2 slots hit all but the first 2 accesses.
pub fn alternation() -> Result<(), String> {
    let trace = decode_only(2, &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    expect("lru 1 slot", hits(&run(&trace, policy("lru")?, 1, 0)?), 0)?;
    expect("lru 2 slots", hits(&run(&trace, policy("lru")?, 2, 0)?), 8)
}

/// Expert 0 every third step between fresh experts: 0, 1, 2, 0, 3, 4, ..., 0, 9, 10, with 2
/// slots. LRU evicts 0 at every third step and never hits. LFU loses 0 at step 2 (counts tie at
/// 1, lower id first), then keeps it from step 3 because its count leads: hits at steps 6, 9,
/// and 12. Belady keeps 0 from step 2 on: hits at steps 3, 6, 9, and 12.
pub fn frequency() -> Result<(), String> {
    let mut ids = Vec::new();
    for i in 0..5u16 {
        ids.extend([0, 1 + 2 * i, 2 + 2 * i]);
    }
    let trace = decode_only(16, &ids);
    expect("lru", hits(&run(&trace, policy("lru")?, 2, 0)?), 0)?;
    expect("lfu", hits(&run(&trace, policy("lfu")?, 2, 0)?), 3)?;
    expect("belady", hits(&run(&trace, policy("belady")?, 2, 0)?), 4)
}

/// Expert 0 four times, then 1 and 2 alternate 4 times each, with 2 slots. Without decay, 0
/// keeps count 4 and 1 and 2 evict each other until step 11: 3 hits. With a half-life of 2
/// admitted steps, the counts halve after steps 1, 3, and 5; at step 5 expert 0 and expert 1 tie
/// at count 1 and 0 leaves, so steps 6 to 11 hit: 9 hits.
pub fn lfu_decay() -> Result<(), String> {
    let trace = decode_only(3, &[0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2]);
    expect("lfu", hits(&run(&trace, policy("lfu")?, 2, 0)?), 3)?;
    let decay = Policy {
        name: "lfu-h2",
        kind: Kind::Lfu,
        half_life: Some(2),
        core: Core::None,
    };
    expect("lfu half-life 2", hits(&run(&trace, decay, 2, 0)?), 9)
}

fn hot(name: &'static str, half_life: u64) -> Policy {
    Policy {
        name,
        kind: Kind::Hot,
        half_life: Some(half_life),
        core: Core::None,
    }
}

/// The hotness of an expert at step `n` is the sum of `2^(-(n - i) / H)` over its uses at the steps
/// `i`, and the victim is the expert with the least hotness, ties to the least recent use. All
/// cases have 2 slots and one expert per step.
///
/// - Half-life 2 (factor 0.7071 per step), steps 0, 0, 0, 0, 0, 1, 2, 0, 1, 0. Expert 0 hits at
///   steps 1 to 4. Expert 1 enters at step 5. At step 6 the hotness of 0 is the sum of the
///   factors 0.7071 to the powers 2 to 6, which is 1.4053, against 0.7071 for 1: 1 leaves, and 2
///   enters. Step 7 hits 0. At step 8 the hotness of 2 is 0.5, so 2 leaves for 1. Step 9 hits 0.
///   Hits per step: 0, 1, 1, 1, 1, 0, 0, 1, 0, 1 (6). LRU scores 5, because it evicts 0 at step 6.
/// - Half-life 1 (factor 0.5), steps 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2. Steps 1 to 3 hit. At
///   step 5 the hotness of 0 is 1/32 + 1/16 + 1/8 + 1/4 = 15/32, against 1/2 for 1, so 0 leaves
///   for good, and steps 6 to 11 hit: 9 hits. LFU without decay scores 3.
/// - Warm start, ranking 0, 1, 2, half-life 2, steps 2, 0, 1. Both residents have hotness 0 at
///   step 0, and 1 is the less recent warm entry: 1 leaves for 2. Step 1 hits 0. At step 2 the
///   hotness of 2 (0.5) is below that of 0 (0.7071), so 2 leaves for 1. Hits per step: 0, 1, 0.
pub fn hot_decay() -> Result<(), String> {
    let seq = |ids: &[u16], experts: usize| decode_only(experts, ids);
    let first = seq(&[0, 0, 0, 0, 0, 1, 2, 0, 1, 0], 3);
    let got = step_hits(&run(&first, hot("hot-h2", 2), 2, 0)?);
    if got != [0, 1, 1, 1, 1, 0, 0, 1, 0, 1] {
        return Err(format!("hot-h2 hits per step {got:?}"));
    }
    expect(
        "lru on the first sequence",
        hits(&run(&first, policy("lru")?, 2, 0)?),
        5,
    )?;
    let second = seq(&[0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2], 3);
    let got = step_hits(&run(&second, hot("hot-h1", 1), 2, 0)?);
    if got != [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1] {
        return Err(format!("hot-h1 hits per step {got:?}"));
    }
    expect(
        "lfu on the second sequence",
        hits(&run(&second, policy("lfu")?, 2, 0)?),
        3,
    )?;
    let third = seq(&[2, 0, 1], 3);
    let got = step_hits(&run_slots(&third, hot("hot-h2", 2), &[2], 0, true)?);
    if got != [0, 1, 0] {
        return Err(format!("warm hot-h2 hits per step {got:?}"));
    }
    Ok(())
}

/// With half-life 1 the weight of step 1,024 is `2^1024`, which a float cannot hold. The replay
/// stops with an error instead of ranking by a wrong hotness.
pub fn hot_overflow() -> Result<(), String> {
    let ids: Vec<u16> = (0..1030).map(|i| i % 2).collect();
    let trace = decode_only(2, &ids);
    match run(&trace, hot("hot-h1", 1), 1, 0) {
        Err(message) if message.contains("overflows") => Ok(()),
        other => Err(format!("1,030 steps with half-life 1 gave {other:?}")),
    }
}

/// Static with 2 slots is a core of 1 expert and one LRU tail slot. Prompt 0 is 0, 0, 0, 1 and
/// prompt 1 is 1, 1, 1, 2, 3. Held out, prompt 1 gets the core {0} from prompt 0 and hits only in
/// its tail (steps 1 and 2): 2 hits. Prompt 0 gets the core {1} from prompt 1: tail hits at steps
/// 1 and 2 and a core hit at step 3: 3 hits.
pub fn static_held_out() -> Result<(), String> {
    let steps = |ids: &[u16]| ids.iter().map(|&e| (true, vec![e])).collect();
    let trace = synthetic(
        1,
        4,
        1,
        vec![(0, steps(&[0, 0, 0, 1])), (1, steps(&[1, 1, 1, 2, 3]))],
    );
    let rankings = held_out_rankings(&trace);
    for (prompt, want) in [(0, 3), (1, 2)] {
        let config = SimConfig {
            policy: policy("static")?,
            slots: &[2],
            warm: false,
            admit_prompt: true,
            page_cache_bytes: 0,
        };
        let outcome = simulate(&trace, &trace.prompts[prompt], &rankings[prompt], &config)?;
        expect(&format!("static prompt {prompt}"), hits(&outcome), want)?;
    }
    Ok(())
}

/// One LRU slot misses all of 0, 1, 0, 2, 0. A 200-byte page cache of 100-byte experts serves
/// steps 2 and 4 (it evicts 1 at step 3): 200 page-cache bytes and 300 SSD bytes.
pub fn page_cache() -> Result<(), String> {
    let trace = decode_only(3, &[0, 1, 0, 2, 0]);
    let ranked = vec![vec![0, 1, 2]];
    let config = SimConfig {
        policy: policy("lru")?,
        slots: &[1],
        warm: false,
        admit_prompt: true,
        page_cache_bytes: 200,
    };
    let outcome = simulate(&trace, &trace.prompts[0], &ranked, &config)?;
    let page_cache: u64 = outcome.decode.iter().map(|s| s.page_cache_bytes).sum();
    let ssd: u64 = outcome.decode.iter().map(|s| s.ssd_bytes).sum();
    if (page_cache, ssd) == (200, 300) {
        Ok(())
    } else {
        Err(format!(
            "page cache {page_cache} B, SSD {ssd} B, expected 200 and 300"
        ))
    }
}

/// Prompt steps 2, 3 and generated steps 2, 3, 2 with 2 LRU slots. Admitted, the prompt fills the
/// cache and every generated step hits (3). Bypassed, the generated steps start empty and only
/// the last hits (1).
pub fn prompt_bypass() -> Result<(), String> {
    let steps = vec![
        (false, vec![2]),
        (false, vec![3]),
        (true, vec![2]),
        (true, vec![3]),
        (true, vec![2]),
    ];
    let trace = synthetic(1, 4, 1, vec![(0, steps)]);
    let ranked = vec![vec![0, 1, 2, 3]];
    for (admit_prompt, want) in [(true, 3), (false, 1)] {
        let config = SimConfig {
            policy: policy("lru")?,
            slots: &[2],
            warm: false,
            admit_prompt,
            page_cache_bytes: 0,
        };
        let outcome = simulate(&trace, &trace.prompts[0], &ranked, &config)?;
        let decode: u64 = outcome.decode.iter().map(|s| u64::from(s.hits)).sum();
        expect(&format!("admit {admit_prompt}"), decode, want)?;
    }
    Ok(())
}

/// Hits of each generated step.
fn step_hits(outcome: &Outcome) -> Vec<u32> {
    outcome.decode.iter().map(|s| s.hits).collect()
}

/// Two layers, two experts, one expert per layer and step. Token `t` selects `[layer 0, layer 1]`.
fn two_layers() -> Trace {
    let steps = [[0, 0], [1, 1], [0, 0], [1, 1]]
        .iter()
        .map(|ids| (true, ids.to_vec()))
        .collect();
    synthetic(2, 2, 1, vec![(0, steps)])
}

/// The pool of 3 slots (slots 2 and 1) over `two_layers`. Keys are the last-use token plus 1.
/// - Token 0: `L0E0` and `L1E0` miss and enter (keys 1, 1). 0 hits.
/// - Token 1: `L0E1` misses and enters (key 2). `L1E1` misses at layer 1 with the pool full. The
///   oldest keys are 1, on `L0E0` and `L1E0`. Layer 1 is in service, so the order is layer 1
///   before layer 0: `L1E0` leaves. `L1E1` enters (key 2). 0 hits.
/// - Token 2: `L0E0` hits (key 3). `L1E0` misses. The oldest keys are 2, on `L0E1` and `L1E1`.
///   `L1E1` leaves and `L1E0` enters (key 3). 1 hit.
/// - Token 3: `L0E1` hits (key 4). `L1E1` misses. The oldest keys are 3, on `L0E0` and `L1E0`.
///   `L1E0` leaves. 1 hit.
///
/// So the hits per token are 0, 0, 1, 1.
pub fn pool_least_stale() -> Result<(), String> {
    let trace = two_layers();
    let outcome = run_slots(&trace, policy("pool")?, &[2, 1], 0, false)?;
    let got = step_hits(&outcome);
    if got == [0, 0, 1, 1] {
        Ok(())
    } else {
        Err(format!(
            "pool hits per token {got:?}, expected [0, 0, 1, 1]"
        ))
    }
}

/// Two layers, 2 experts, 2 per step, 3 slots (2 and 1), and the selection `[0, 1]` in both layers
/// at tokens 0, 1, and 2. Keys are the last-use token plus 1.
/// - Token 0, layer 0: `L0E0` and `L0E1` miss and enter (key 1). Layer 1: `L1E0` misses and
///   enters (key 1). `L1E1` misses with the pool full. The order for layer 1 is layer 1, then
///   layer 0 with the lower id first. `L1E0` is in the step, so it stays, and `L0E0` leaves.
///   `L1E1` enters. The pool holds `L0E1`, `L1E0`, and `L1E1`, all with key 1. 0 hits.
/// - Token 1, layer 0: `L0E1` hits (key 2). `L0E0` misses with the pool full. The oldest keys are
///   1, on `L1E0` and `L1E1`. The order for layer 0 is layer 0, then layer 1 with the lower id
///   first, so `L1E0` leaves and `L0E0` enters (key 2). 1 hit. Layer 1: `L1E1` hits (key 2).
///   `L1E0` misses with the pool full and every key 2. `L1E1` is in the step, so it stays, and
///   `L0E0` leaves. `L1E0` enters. 1 hit.
/// - Token 2 repeats token 1: 2 hits.
///
/// So the hits per token are 0, 2, 2.
pub fn pool_keeps_the_step() -> Result<(), String> {
    let steps = (0..3).map(|_| (true, vec![0, 1, 0, 1])).collect();
    let trace = synthetic(2, 2, 2, vec![(0, steps)]);
    let got = step_hits(&run_slots(&trace, policy("pool")?, &[2, 1], 0, false)?);
    if got == [0, 2, 2] {
        Ok(())
    } else {
        Err(format!("pool hits per token {got:?}, expected [0, 2, 2]"))
    }
}

/// Three layers, 3 experts, 2 per step, a warm pool of 3 slots (1 per layer), and one token that
/// selects `[0, 1]` in layer 0, `[0, 2]` in layer 1, and `[1, 2]` in layer 2. The warm pool holds
/// `L0E0`, `L1E0`, and `L2E0`, all with key 0.
/// - Layer 0 is in service. `L0E0` hits (key 1). `L0E1` misses with the pool full. The oldest
///   key is 0, on `L1E0` and `L2E0`. Layer 0 has no item with key 0 outside the step, so the
///   order goes on with the layers above it, from the last layer down: layer 2 before layer 1.
///   `L2E0` leaves. `L0E1` enters.
/// - Layer 1: `L1E0` hits, because it stayed. 1 hit.
/// - Layer 2: `L2E1` and `L2E2` miss, and `L2E0` is not selected. 0 hits.
///
/// So the token has 1 + 1 + 0 = 2 hits. If the layers above came in the order 1, 2, `L1E0` would
/// leave, and the token would have 1 hit.
pub fn pool_ties_above_the_served_layer() -> Result<(), String> {
    let trace = synthetic(3, 3, 2, vec![(0, vec![(true, vec![0, 1, 0, 2, 1, 2])])]);
    let got = step_hits(&run_slots(&trace, policy("pool")?, &[1, 1, 1], 0, true)?);
    if got == [2] {
        Ok(())
    } else {
        Err(format!("pool hits per token {got:?}, expected [2]"))
    }
}

/// One layer, 3 experts, 2 slots, warm start with the ranking 0, 1, 2: the pool holds experts 0
/// and 1, both stale since token -1. Token 0 selects 2: it misses, and the tie goes to the lower
/// expert id, so 0 leaves. Token 1 selects 1: it hits. So the hits per token are 0, 1.
///
/// The pooled Belady starts from the same experts, keyed by their first use. On the steps 2, 1, 0
/// the first uses are 1 at token 1 and 0 at token 2. Token 0 selects 2: it misses, and 0 has the
/// furthest next use, so 0 leaves. Token 1 selects 1: it hits. Token 2 selects 0: it misses. So the
/// hits per token are 0, 1, 0.
pub fn pool_warm_start() -> Result<(), String> {
    let steps = [2, 1].iter().map(|&e| (true, vec![e])).collect();
    let trace = synthetic(1, 3, 1, vec![(0, steps)]);
    let got = step_hits(&run_slots(&trace, policy("pool")?, &[2], 0, true)?);
    if got != [0, 1] {
        return Err(format!("warm pool hits per token {got:?}, expected [0, 1]"));
    }
    let steps = [2, 1, 0].iter().map(|&e| (true, vec![e])).collect();
    let trace = synthetic(1, 3, 1, vec![(0, steps)]);
    let got = step_hits(&run_slots(&trace, policy("belady-pool")?, &[2], 0, true)?);
    if got == [0, 1, 0] {
        Ok(())
    } else {
        Err(format!(
            "warm pooled Belady hits per token {got:?}, expected [0, 1, 0]"
        ))
    }
}

/// One layer, 3 experts, a pool of 1 slot. The prompt steps select 0 and 1, and the generated steps
/// select 1 and 1. With admission, the prompt leaves 1 in the slot, so both generated steps hit:
/// hits per token 1, 1. With bypass, the prompt steps change nothing, so the first generated step
/// misses and the second hits: 0, 1.
pub fn pool_prompt_bypass() -> Result<(), String> {
    let steps = vec![
        (false, vec![0]),
        (false, vec![1]),
        (true, vec![1]),
        (true, vec![1]),
    ];
    let trace = synthetic(1, 3, 1, vec![(0, steps)]);
    let ranked = vec![vec![0, 1, 2]];
    for (admit_prompt, want) in [(true, [1, 1]), (false, [0, 1])] {
        let config = SimConfig {
            policy: policy("pool")?,
            slots: &[1],
            warm: false,
            admit_prompt,
            page_cache_bytes: 0,
        };
        let got = step_hits(&simulate(&trace, &trace.prompts[0], &ranked, &config)?);
        if got != want {
            return Err(format!(
                "pool admit {admit_prompt}: hits per token {got:?}, expected {want:?}"
            ));
        }
    }
    Ok(())
}

/// The `two_layers` trace with the pooled Belady and 3 slots. Times are `token * 2 + layer`.
/// - Token 0: `L0E0` (next use time 4) and `L1E0` (5) miss and enter.
/// - Token 1: `L0E1` (6) misses and enters. `L1E1` (7) misses with the pool full. `L0E1` has the
///   furthest next use among the items outside the step, so it leaves. 0 hits.
/// - Token 2: `L0E0` and `L1E0` hit and are never used again: 2 hits.
/// - Token 3: `L0E1` misses. `L0E0` and `L1E0` are never used again and `L1E1` is used at once;
///   the tie goes to the lower layer, so `L0E0` leaves. `L1E1` hits: 1 hit.
///
/// So the hits per token are 0, 0, 2, 1 (3 in all). The per-layer Belady with the same 3 slots
/// (2 and 1) scores 2: layer 0 hits at tokens 2 and 3, and the 1-slot layer 1 never hits. So the
/// per-layer Belady is not a bound of the pool.
pub fn pool_belady() -> Result<(), String> {
    let trace = two_layers();
    let pooled = run_slots(&trace, policy("belady-pool")?, &[2, 1], 0, false)?;
    let got = step_hits(&pooled);
    if got != [0, 0, 2, 1] {
        return Err(format!(
            "pooled Belady hits per token {got:?}, expected [0, 0, 2, 1]"
        ));
    }
    let layered = run_slots(&trace, policy("belady")?, &[2, 1], 0, false)?;
    expect("per-layer belady, slots 2 and 1", hits(&layered), 2)?;
    // One layer, 2 experts per step, 2 slots: [0, 1], [1, 2], [0, 2]. Token 1 hits 1 and misses 2.
    // The pool must evict 0 (outside the step) although 1 (in the step) is never used again, so
    // token 2 hits 2 and misses 0. The hits per token are 0, 1, 1.
    let steps = [[0, 1], [1, 2], [0, 2]]
        .iter()
        .map(|ids| (true, ids.to_vec()))
        .collect();
    let trace = synthetic(1, 3, 2, vec![(0, steps)]);
    let got = step_hits(&run_slots(&trace, policy("belady-pool")?, &[2], 0, false)?);
    if got == [0, 1, 1] {
        Ok(())
    } else {
        Err(format!("hits per token {got:?}, expected [0, 1, 1]"))
    }
}

/// Two layers with experts of 100 and 300 bytes, one expert per layer and step, the page-cache
/// baseline with the slots 1 and 1, so 100 + 300 = 400 bytes. The steps select `[layer 0, layer 1]`:
/// `[0, 0]`, `[0, 1]`, `[1, 1]`, `[0, 1]`. Items are `L0E0`, `L1E0` (300 bytes), and so on.
/// - Step 0: both miss and enter (400 bytes used). Page-cache bytes 0, SSD bytes 400.
/// - Step 1: `L0E0` hits (now the newest). `L1E1` misses and needs 300 bytes: the oldest item is
///   `L1E0`, so it leaves (100 used) and `L1E1` enters. Page-cache bytes 100, SSD bytes 300.
/// - Step 2: `L0E1` misses and needs 100 bytes: the oldest item is `L0E0`, so it leaves. `L1E1`
///   hits. Page-cache bytes 300, SSD bytes 100.
/// - Step 3: `L0E0` misses; the oldest item is `L0E1`, so it leaves. `L1E1` hits. Page-cache
///   bytes 300, SSD bytes 100.
///
/// So the hits per step are 0, 1, 1, 1; the page-cache bytes are 0, 100, 300, 300; and the SSD
/// bytes are 400, 300, 100, 100.
pub fn page_only() -> Result<(), String> {
    let steps = [[0, 0], [0, 1], [1, 1], [0, 1]]
        .iter()
        .map(|ids| (true, ids.to_vec()))
        .collect();
    let mut trace = synthetic(2, 2, 1, vec![(0, steps)]);
    trace.expert_bytes = vec![100, 300];
    let outcome = run_slots(&trace, policy("page-only")?, &[1, 1], 0, false)?;
    let got = (
        step_hits(&outcome),
        outcome
            .decode
            .iter()
            .map(|s| s.page_cache_bytes)
            .collect::<Vec<_>>(),
        outcome
            .decode
            .iter()
            .map(|s| s.ssd_bytes)
            .collect::<Vec<_>>(),
    );
    let want = (
        vec![0, 1, 1, 1],
        vec![0, 100, 300, 300],
        vec![400, 300, 100, 100],
    );
    if got == want {
        Ok(())
    } else {
        Err(format!("page-only {got:?}, expected {want:?}"))
    }
}

/// The decision value `h*` takes the per-layer candidates only: the bounds and the pool stay out.
/// The pool has the pooled Belady as its bound, every other policy the per-layer Belady.
pub fn candidates_and_bounds() -> Result<(), String> {
    let names: Vec<&str> = crate::cache::POLICIES
        .iter()
        .filter(|p| p.kind.is_candidate())
        .map(|p| p.name)
        .collect();
    let want = [
        "lru",
        "lfu",
        "lfu-h64",
        "lfu-h256",
        "lfu-h1024",
        "hot-h16",
        "hot-h64",
        "hot-h256",
        "static",
        "static-half",
    ];
    if names != want {
        return Err(format!("candidates {names:?}, expected {want:?}"));
    }
    if Kind::Pool.bound() != Kind::PoolBelady
        || Kind::Lru.bound() != Kind::Belady
        || Kind::PageOnly.bound() != Kind::PoolBelady
        || Kind::Hot.bound() != Kind::Belady
        || Kind::Static.bound() != Kind::Belady
    {
        return Err("the bound of the pool is the pooled Belady, of the rest the Belady".into());
    }
    Ok(())
}

/// A seeded skewed trace: 2 layers, 16 experts, 3 per step, 2 prompts of 40 prompt and 60
/// generated steps.
pub fn seeded() -> Trace {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut draw = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let mut prompts = Vec::new();
    for id in 0..2u32 {
        let mut steps = Vec::new();
        for s in 0..100 {
            let mut sel = Vec::new();
            for _layer in 0..2 {
                let mut picked: Vec<u16> = Vec::new();
                while picked.len() < 3 {
                    // Squaring a uniform draw skews the ids toward 0.
                    let u = f64::from(draw() % 10_000) / 10_000.0;
                    let e = (u * u * 16.0) as u16;
                    if !picked.contains(&e) {
                        picked.push(e);
                    }
                }
                sel.extend(picked);
            }
            steps.push((s >= 40, sel));
        }
        prompts.push((id, steps));
    }
    synthetic(2, 16, 3, prompts)
}

/// `trace` as the engine writes `trace.tsv`, with probabilities and gates of `1 / top_k`.
#[cfg(test)]
pub fn to_tsv(trace: &Trace) -> String {
    let k = trace.top_k;
    let bytes: Vec<String> = trace.expert_bytes.iter().map(u64::to_string).collect();
    let mut out = format!(
        "# schema={}\n# layers={}\n# experts={}\n# top_k={k}\n# renorm=true\n# expert_bytes={}\n{}\n",
        crate::trace::SCHEMA,
        trace.layers,
        trace.experts,
        bytes.join(","),
        crate::trace::columns_line(k)
    );
    let share = format!("{:.8e}", 1.0 / k as f64);
    for prompt in &trace.prompts {
        for (pos, step) in prompt.steps.iter().enumerate() {
            for layer in 0..trace.layers {
                let ids: Vec<String> = trace.sel(step, layer).iter().map(u16::to_string).collect();
                let values = vec![share.as_str(); 2 * k].join("\t");
                out.push_str(&format!(
                    "{}\t{}\t{pos}\t{layer}\t{}\t{values}\n",
                    prompt.id,
                    if step.decode { "g" } else { "p" },
                    ids.join("\t")
                ));
            }
        }
    }
    out
}

/// On the seeded trace, Belady has at least the hits of every other policy from the same start at
/// every size, and the LRU hits from stack distances equal the direct LRU replay at 3 sizes.
pub fn belady_bound_and_lru_curve() -> Result<(), String> {
    let trace = seeded();
    let rankings = held_out_rankings(&trace);
    for slots in 6..=16 {
        for warm in [false, true] {
            let total = |p: Policy| -> Result<u64, String> {
                let mut sum = 0;
                for (i, prompt) in trace.prompts.iter().enumerate() {
                    let config = SimConfig {
                        policy: p,
                        slots: &[slots, slots],
                        warm,
                        admit_prompt: true,
                        page_cache_bytes: 0,
                    };
                    sum += hits(&simulate(&trace, prompt, &rankings[i], &config)?);
                }
                Ok(sum)
            };
            let bound = total(policy("belady")?)?;
            let pool_bound = total(policy("belady-pool")?)?;
            for p in crate::cache::POLICIES {
                // An empty start of a static policy still holds its core: another start state.
                if matches!(p.kind, Kind::Belady | Kind::PoolBelady)
                    || (p.kind == Kind::Static && !warm)
                {
                    continue;
                }
                if p.kind == Kind::Static && slots - p.core_size(slots, 3) < 3 {
                    continue;
                }
                let got = total(p)?;
                let limit = if p.kind.bound() == Kind::PoolBelady {
                    pool_bound
                } else {
                    bound
                };
                if got > limit {
                    return Err(format!(
                        "{} with {slots} slots (warm {warm}): {got} hits above Belady {limit}",
                        p.name
                    ));
                }
            }
        }
    }
    for slots in [3, 7, 12] {
        let mut direct = 0;
        for prompt in 0..trace.prompts.len() {
            direct += hits(&run(&trace, policy("lru")?, slots, prompt)?);
        }
        let from_distances = lru_hits_from_distances(&trace, slots);
        if direct != from_distances {
            return Err(format!(
                "{slots} slots: direct LRU {direct} hits, stack distances {from_distances}"
            ));
        }
    }
    Ok(())
}

/// T0-G1 and T0-G2 with hand-derived values: `miss_bytes / bw` is 0.2654208 s at 5e9 B/s and
/// 0.442368 s at 3e9 B/s for 1,327,104,000 B. So G1 is 0.149 + 0.1 * 0.2654208 = 0.17554208 s
/// (go) and 0.149 + 0.5 * 0.442368 = 0.370184 s (no).
pub fn gate_values() -> Result<(), String> {
    let miss = 1_327_104_000.0;
    let go = gates(0.9, 0.149, miss, 5e9, 0.377);
    let no = gates(0.5, 0.149, miss, 3e9, 0.377);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9;
    let ok = close(go.g1_seconds, 0.175_542_08)
        && go.g1
        && close(go.g2_saved, 0.238_878_72)
        && go.g2
        && close(no.g1_seconds, 0.370_184)
        && !no.g1
        && close(no.g2_saved, 0.221_184)
        && !no.g2;
    if ok {
        Ok(())
    } else {
        Err(format!("gates {go:?} and {no:?}"))
    }
}

/// `1 2 3 4 1 2 3 4` has 5 4-grams and repeats one: 0.2. Nearest-rank percentiles of
/// `5 1 4 2 3`: p50 is 3 and p95 is 5.
pub fn repetition_and_percentiles() -> Result<(), String> {
    let r = repetition(&[1, 2, 3, 4, 1, 2, 3, 4]);
    let p = [5, 1, 4, 2, 3];
    if (r - 0.2).abs() <= 1e-12 && percentile(&p, 0.5) == 3 && percentile(&p, 0.95) == 5 {
        Ok(())
    } else {
        Err(format!(
            "repetition {r}, p50 {}, p95 {}",
            percentile(&p, 0.5),
            percentile(&p, 0.95)
        ))
    }
}

/// A two-layer trace text whose second step lacks layer 1 fails the parse (T0-P1), and the same
/// text with the row present parses to the expected selections.
pub fn parse_checks() -> Result<(), String> {
    let head = "# schema=s3-route-trace/1\n# layers=2\n# experts=4\n# top_k=2\n# renorm=true\n\
                # expert_bytes=10,20\nprompt\tmode\tpos\tlayer\te0\te1\tp0\tp1\tg0\tg1\n";
    let row = |mode: &str, pos: u32, layer: u32, a: u16, b: u16| {
        format!("7\t{mode}\t{pos}\t{layer}\t{a}\t{b}\t5e-1\t2.5e-1\t6.66666687e-1\t3.33333343e-1\n")
    };
    let complete = format!(
        "{head}{}{}{}{}",
        row("p", 0, 0, 2, 0),
        row("p", 0, 1, 1, 3),
        row("g", 1, 0, 3, 2),
        row("g", 1, 1, 0, 1)
    );
    let trace = Trace::parse(&complete)?;
    let want = synthetic(
        2,
        4,
        2,
        vec![(7, vec![(false, vec![2, 0, 1, 3]), (true, vec![3, 2, 0, 1])])],
    );
    if (&trace.prompts, &trace.expert_bytes) != (&want.prompts, &vec![10, 20]) {
        return Err(format!("parsed {:?}", trace.prompts));
    }
    let missing = format!(
        "{head}{}{}{}",
        row("p", 0, 0, 2, 0),
        row("p", 0, 1, 1, 3),
        row("g", 1, 0, 3, 2)
    );
    match Trace::parse(&missing) {
        Err(message) if message.contains("1 of 2 layers") => Ok(()),
        other => Err(format!("a missing layer parsed as {other:?}")),
    }
}

/// Every case by name.
pub fn cases() -> Vec<(&'static str, Case)> {
    vec![
        ("lru_cycle", lru_cycle),
        ("belady_cycle", belady_cycle),
        ("alternation", alternation),
        ("frequency", frequency),
        ("lfu_decay", lfu_decay),
        ("hot_decay", hot_decay),
        ("hot_overflow", hot_overflow),
        ("static_held_out", static_held_out),
        ("candidates_and_bounds", candidates_and_bounds),
        ("pool_least_stale", pool_least_stale),
        ("pool_warm_start", pool_warm_start),
        ("pool_keeps_the_step", pool_keeps_the_step),
        (
            "pool_ties_above_the_served_layer",
            pool_ties_above_the_served_layer,
        ),
        ("pool_prompt_bypass", pool_prompt_bypass),
        ("pool_belady", pool_belady),
        ("page_only", page_only),
        ("page_cache", page_cache),
        ("prompt_bypass", prompt_bypass),
        ("belady_bound_and_lru_curve", belady_bound_and_lru_curve),
        ("gate_values", gate_values),
        ("repetition_and_percentiles", repetition_and_percentiles),
        ("parse_checks", parse_checks),
    ]
}

/// Runs every case.
pub fn run_all() -> Vec<(&'static str, Result<(), String>)> {
    cases()
        .into_iter()
        .map(|(name, case)| (name, case()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_scores_zero_on_a_cycle_one_longer_than_the_cache() {
        assert_eq!(lru_cycle(), Ok(()));
    }

    #[test]
    fn belady_keeps_the_experts_used_soonest() {
        assert_eq!(belady_cycle(), Ok(()));
    }

    #[test]
    fn two_experts_alternate() {
        assert_eq!(alternation(), Ok(()));
    }

    #[test]
    fn lfu_keeps_a_frequent_expert_that_lru_evicts() {
        assert_eq!(frequency(), Ok(()));
    }

    #[test]
    fn lfu_decay_halves_every_count() {
        assert_eq!(lfu_decay(), Ok(()));
    }

    #[test]
    fn hot_keeps_the_recently_hot_expert_and_decays_the_old_one() {
        assert_eq!(hot_decay(), Ok(()));
    }

    #[test]
    fn hot_stops_when_the_scaled_hotness_overflows() {
        assert_eq!(hot_overflow(), Ok(()));
    }

    #[test]
    fn static_cores_come_from_the_other_prompts() {
        assert_eq!(static_held_out(), Ok(()));
    }

    #[test]
    fn the_decision_value_takes_only_per_layer_candidates_and_bounds_follow_the_design() {
        assert_eq!(candidates_and_bounds(), Ok(()));
    }

    #[test]
    fn the_pool_evicts_the_stalest_item_and_the_layer_that_comes_up_last() {
        assert_eq!(pool_least_stale(), Ok(()));
    }

    #[test]
    fn the_pool_breaks_a_tie_above_the_served_layer_at_the_last_layer() {
        assert_eq!(pool_ties_above_the_served_layer(), Ok(()));
    }

    #[test]
    fn a_warm_pool_holds_the_ranked_experts_of_every_layer() {
        assert_eq!(pool_warm_start(), Ok(()));
    }

    #[test]
    fn the_pool_never_evicts_an_expert_of_the_step() {
        assert_eq!(pool_keeps_the_step(), Ok(()));
    }

    #[test]
    fn the_pooled_belady_beats_the_per_layer_belady() {
        assert_eq!(pool_belady(), Ok(()));
    }

    #[test]
    fn the_page_cache_baseline_serves_every_read_by_bytes() {
        assert_eq!(page_only(), Ok(()));
    }

    #[test]
    fn a_bypassed_prompt_leaves_the_pool_unchanged() {
        assert_eq!(pool_prompt_bypass(), Ok(()));
    }

    #[test]
    fn the_page_cache_serves_repeated_misses() {
        assert_eq!(page_cache(), Ok(()));
    }

    #[test]
    fn bypassed_prompts_leave_the_cache_unchanged() {
        assert_eq!(prompt_bypass(), Ok(()));
    }

    #[test]
    fn belady_bounds_every_policy_and_stack_distances_give_lru() {
        assert_eq!(belady_bound_and_lru_curve(), Ok(()));
    }

    #[test]
    fn gates_follow_the_plan_formulas() {
        assert_eq!(gate_values(), Ok(()));
    }

    #[test]
    fn repetition_and_percentiles_match_hand_values() {
        assert_eq!(repetition_and_percentiles(), Ok(()));
    }

    #[test]
    fn the_parser_rejects_a_missing_layer() {
        assert_eq!(parse_checks(), Ok(()));
    }
}
