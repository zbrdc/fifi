//! Joint replay of all layers of one prompt with an optional page-cache second level, the
//! training profile, LRU stack distances, and the greedy per-layer split (S3 plan section 3.5).

use crate::cache::{Kind, LayerCache, NEVER, Policy};
use crate::pool::{PoolCache, PoolRule};
use crate::trace::{PromptTrace, Trace};
use std::collections::{BTreeSet, HashMap};

/// One replay of one prompt.
#[derive(Debug, Clone)]
pub struct SimConfig<'a> {
    pub policy: Policy,
    /// Slots of each layer.
    pub slots: &'a [usize],
    /// Start with the slots filled from the training ranking (empty otherwise; a static core is
    /// always loaded).
    pub warm: bool,
    /// Prompt steps admit their misses (`false`: prompt bypass, no cache change in prompt steps).
    pub admit_prompt: bool,
    /// Capacity of the LRU page cache in front of the SSD, in bytes (0: none).
    pub page_cache_bytes: u64,
}

/// Costs of one generated step over all layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeStep {
    pub hits: u32,
    pub misses: u32,
    /// Miss bytes served by the page cache.
    pub page_cache_bytes: u64,
    /// Miss bytes read from the SSD.
    pub ssd_bytes: u64,
}

/// The result of one replay.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// Generated steps in order.
    pub decode: Vec<DecodeStep>,
    pub prompt_hits: u64,
    pub prompt_accesses: u64,
    /// Generated-step hits and accesses of each layer.
    pub layer_hits: Vec<u64>,
    pub layer_accesses: Vec<u64>,
}

/// An LRU page cache of whole experts, sized in bytes, shared by all layers.
struct PageCache {
    capacity: u64,
    used: u64,
    clock: u64,
    key: HashMap<u32, (u64, u64)>,
    order: BTreeSet<(u64, u32)>,
}

impl PageCache {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            clock: 0,
            key: HashMap::new(),
            order: BTreeSet::new(),
        }
    }

    /// Reads `item` of `bytes`. Returns whether the page cache held it. The item is the most
    /// recent afterwards, unless it is larger than the whole cache.
    fn read(&mut self, item: u32, bytes: u64) -> bool {
        self.clock += 1;
        if let Some((key, size)) = self.key.remove(&item) {
            self.order.remove(&(key, item));
            self.key.insert(item, (self.clock, size));
            self.order.insert((self.clock, item));
            return true;
        }
        if bytes > self.capacity {
            return false;
        }
        while self.used + bytes > self.capacity {
            let Some((_, old)) = self.order.pop_first() else {
                break;
            };
            if let Some((_, size)) = self.key.remove(&old) {
                self.used -= size;
            }
        }
        self.used += bytes;
        self.key.insert(item, (self.clock, bytes));
        self.order.insert((self.clock, item));
        false
    }
}

/// Experts of each layer from the most to the least selected over `prompts` (all steps), ties to
/// the lower id.
pub fn ranking(trace: &Trace, prompts: &[&PromptTrace]) -> Vec<Vec<u16>> {
    (0..trace.layers)
        .map(|layer| {
            let mut counts = vec![0u64; trace.experts];
            for prompt in prompts {
                for step in &prompt.steps {
                    for &e in trace.sel(step, layer) {
                        counts[usize::from(e)] += 1;
                    }
                }
            }
            let mut ranked: Vec<u16> = (0..trace.experts as u16).collect();
            ranked.sort_by_key(|&e| (std::cmp::Reverse(counts[usize::from(e)]), e));
            ranked
        })
        .collect()
}

/// For each admitted step and rank of `layer`, the next admitted step index that selects the same
/// expert, or `NEVER`; and each expert's first admitted step, or `NEVER`.
fn next_uses(
    trace: &Trace,
    prompt: &PromptTrace,
    layer: usize,
    admitted: &[usize],
) -> (Vec<u32>, Vec<u32>) {
    let k = trace.top_k;
    let mut next = vec![NEVER; admitted.len() * k];
    let mut seen = vec![NEVER; trace.experts];
    for (index, &step) in admitted.iter().enumerate().rev() {
        let sel = trace.sel(&prompt.steps[step], layer);
        for (rank, &e) in sel.iter().enumerate() {
            next[index * k + rank] = seen[usize::from(e)];
        }
        for &e in sel {
            seen[usize::from(e)] = index as u32;
        }
    }
    (next, seen)
}

/// Replays `prompt` under `config`. `ranked` is the training ranking of each layer.
pub fn simulate(
    trace: &Trace,
    prompt: &PromptTrace,
    ranked: &[Vec<u16>],
    config: &SimConfig<'_>,
) -> Result<Outcome, String> {
    let (layers, k) = (trace.layers, trace.top_k);
    if config.slots.len() != layers || ranked.len() != layers {
        return Err("slots or ranking do not cover every layer".into());
    }
    let admitted: Vec<usize> = (0..prompt.steps.len())
        .filter(|&s| config.admit_prompt || prompt.steps[s].decode)
        .collect();
    let rule = match config.policy.kind {
        Kind::Pool => Some(PoolRule::LeastStale),
        Kind::PoolBelady => Some(PoolRule::Belady),
        _ => None,
    };
    let page_only = config.policy.kind == Kind::PageOnly;
    let mut caches = Vec::with_capacity(layers);
    let mut next = Vec::with_capacity(layers);
    let mut firsts = Vec::with_capacity(layers);
    for (layer, (&slots, layer_ranked)) in config.slots.iter().zip(ranked).enumerate() {
        let (layer_next, first) = if matches!(config.policy.kind, Kind::Belady | Kind::PoolBelady) {
            next_uses(trace, prompt, layer, &admitted)
        } else {
            (Vec::new(), vec![NEVER; trace.experts])
        };
        if rule.is_none() && !page_only {
            caches.push(LayerCache::new(
                config.policy,
                trace.experts,
                slots,
                k,
                layer_ranked,
                config.warm,
                &first,
            )?);
        }
        next.push(layer_next);
        firsts.push(first);
    }
    let mut pool = rule
        .map(|rule| {
            PoolCache::new(
                rule,
                trace.experts,
                config.slots,
                k,
                ranked,
                config.warm,
                &firsts,
            )
        })
        .transpose()?;
    // The page-cache baseline holds as many bytes as the slots of the other policies.
    let page_cache_bytes = if page_only {
        config
            .slots
            .iter()
            .zip(&trace.expert_bytes)
            .map(|(&s, &b)| s as u64 * b)
            .sum()
    } else {
        config.page_cache_bytes
    };
    let mut page_cache = PageCache::new(page_cache_bytes);
    let mut outcome = Outcome {
        layer_hits: vec![0; layers],
        layer_accesses: vec![0; layers],
        ..Outcome::default()
    };
    let mut misses = Vec::with_capacity(k);
    let mut admitted_index = 0usize;
    for step in &prompt.steps {
        let admit = config.admit_prompt || step.decode;
        let mut cost = DecodeStep::default();
        for layer in 0..layers {
            let sel = trace.sel(step, layer);
            let next_sel: &[u32] = if admit && !next[layer].is_empty() {
                &next[layer][admitted_index * k..(admitted_index + 1) * k]
            } else {
                &[]
            };
            let bytes = trace.expert_bytes[layer];
            let hits = if page_only {
                // No resident cache: every read goes through the page cache, and prompt steps
                // fill it too, as the operating system does.
                misses.clear();
                let mut hits = 0;
                for &e in sel {
                    if page_cache.read((layer * trace.experts + usize::from(e)) as u32, bytes) {
                        hits += 1;
                        cost.page_cache_bytes += bytes;
                    } else {
                        misses.push(e);
                        cost.ssd_bytes += bytes;
                    }
                }
                hits
            } else {
                let hits = match pool.as_mut() {
                    Some(pool) => pool.access(
                        admitted_index as u32,
                        layer,
                        sel,
                        admit,
                        next_sel,
                        &mut misses,
                    )?,
                    None => caches[layer].access(sel, admit, next_sel, &mut misses)?,
                };
                for &e in &misses {
                    let item = (layer * trace.experts + usize::from(e)) as u32;
                    if page_cache.read(item, bytes) {
                        cost.page_cache_bytes += bytes;
                    } else {
                        cost.ssd_bytes += bytes;
                    }
                }
                hits
            };
            if step.decode {
                outcome.layer_hits[layer] += u64::from(hits);
                outcome.layer_accesses[layer] += k as u64;
            }
            cost.hits += hits;
            cost.misses += misses.len() as u32;
        }
        if admit {
            admitted_index += 1;
        }
        if step.decode {
            outcome.decode.push(cost);
        } else {
            outcome.prompt_hits += u64::from(cost.hits);
            outcome.prompt_accesses += (layers * k) as u64;
        }
    }
    Ok(outcome)
}

/// A distance for an expert that was never used before.
pub const COLD: u32 = u32::MAX;

/// LRU stack distances of every access of `layer` in `prompt`, in step and rank order, with every
/// step admitted. An access hits an LRU cache of `S` slots exactly when its distance is below
/// `S`: the distance counts the experts whose last use is later than the accessed expert's last
/// use, with ties inside one step ordered by id as the replay orders them.
pub fn stack_distances(trace: &Trace, prompt: &PromptTrace, layer: usize) -> Vec<u32> {
    let e_count = trace.experts;
    let size = prompt.steps.len() * e_count + 1;
    let mut tree = Fenwick::new(size);
    let mut last: Vec<Option<usize>> = vec![None; e_count];
    let mut out = Vec::with_capacity(prompt.steps.len() * trace.top_k);
    let mut live = 0u64;
    for (t, step) in prompt.steps.iter().enumerate() {
        let sel = trace.sel(step, layer);
        for &e in sel {
            out.push(match last[usize::from(e)] {
                None => COLD,
                Some(position) => (live - tree.prefix(position + 1)) as u32,
            });
        }
        for &e in sel {
            let position = t * e_count + usize::from(e);
            if let Some(old) = last[usize::from(e)] {
                tree.add(old, -1);
            } else {
                live += 1;
            }
            tree.add(position, 1);
            last[usize::from(e)] = Some(position);
        }
    }
    out
}

/// A Fenwick tree of counts.
struct Fenwick {
    tree: Vec<i64>,
}

impl Fenwick {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; size + 1],
        }
    }

    fn add(&mut self, index: usize, delta: i64) {
        let mut i = index + 1;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += i.isolate_lowest_one();
        }
    }

    /// The sum of the first `count` positions.
    fn prefix(&self, count: usize) -> u64 {
        let mut i = count;
        let mut sum = 0i64;
        while i > 0 {
            sum += self.tree[i];
            i -= i.isolate_lowest_one();
        }
        sum as u64
    }
}

/// Splits `budget` bytes over layers greedily. Each layer starts at `min` slots. Each next slot
/// goes to the layer with the largest `gain(layer, current slots)` per byte that still fits, ties
/// to the lower layer. A layer stops at `max` slots.
pub fn greedy_split(
    bytes: &[u64],
    budget: u64,
    min: usize,
    max: usize,
    gain: impl Fn(usize, usize) -> u64,
) -> Vec<usize> {
    let mut slots = vec![min; bytes.len()];
    let mut left = budget.saturating_sub(bytes.iter().map(|b| b * min as u64).sum());
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (layer, &b) in bytes.iter().enumerate() {
            if slots[layer] >= max || b > left {
                continue;
            }
            let value = gain(layer, slots[layer]) as f64 / b as f64;
            if best.is_none_or(|(_, v)| value > v) {
                best = Some((layer, value));
            }
        }
        let Some((layer, _)) = best else {
            break;
        };
        slots[layer] += 1;
        left -= bytes[layer];
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Policy;
    use crate::selftest::synthetic;

    /// Two layers with 10 and 20 bytes per expert, four experts, one expert per step. The first
    /// step is a prompt step. Layer 0 selects 0, 0, 1, 0 and layer 1 selects 0, 1, 2, 2.
    fn two_layers() -> Trace {
        let mut trace = synthetic(
            2,
            4,
            1,
            vec![(
                0,
                vec![
                    (false, vec![0, 0]),
                    (true, vec![0, 1]),
                    (true, vec![1, 2]),
                    (true, vec![0, 2]),
                ],
            )],
        );
        trace.expert_bytes = vec![10, 20];
        trace
    }

    fn replay(slots: usize, page_cache_bytes: u64) -> Outcome {
        let trace = two_layers();
        let ranked = vec![vec![0, 1, 2, 3]; 2];
        let config = SimConfig {
            policy: Policy::named("lru").unwrap(),
            slots: &[slots, slots],
            warm: false,
            admit_prompt: true,
            page_cache_bytes,
        };
        simulate(&trace, &trace.prompts[0], &ranked, &config).unwrap()
    }

    #[test]
    fn each_layer_counts_its_own_generated_hits_and_bytes() {
        // Two LRU slots per layer, empty start. Layer 0 (10 bytes per expert): 0 misses in the
        // prompt step, then hits at step 1, misses 1 at step 2 (cache {0, 1}), hits 0 at step 3.
        // Layer 1 (20 bytes): 0 and 1 miss, 2 misses at step 2 and evicts 0, 2 hits at step 3.
        // Generated hits: layer 0 has 2 of 3, layer 1 has 1 of 3.
        // Step 1: layer 0 hits, layer 1 misses 1 = 20 bytes. Step 2: 10 + 20 = 30 bytes.
        // Step 3: two hits, no bytes. The prompt step has 0 hits for 2 accesses.
        let outcome = replay(2, 0);
        assert_eq!(outcome.layer_hits, vec![2, 1]);
        assert_eq!(outcome.layer_accesses, vec![3, 3]);
        assert_eq!(
            outcome.decode,
            vec![
                DecodeStep {
                    hits: 1,
                    misses: 1,
                    page_cache_bytes: 0,
                    ssd_bytes: 20
                },
                DecodeStep {
                    hits: 0,
                    misses: 2,
                    page_cache_bytes: 0,
                    ssd_bytes: 30
                },
                DecodeStep {
                    hits: 2,
                    misses: 0,
                    page_cache_bytes: 0,
                    ssd_bytes: 0
                },
            ]
        );
        assert_eq!((outcome.prompt_hits, outcome.prompt_accesses), (0, 2));
    }

    #[test]
    fn the_page_cache_charges_the_bytes_of_the_missed_layer() {
        // One LRU slot per layer and a page cache of 1000 bytes. Layer 0 selects 0, 0, 1, 0 and
        // layer 1 selects 0, 1, 2, 2. Misses in order: prompt step layer 0 expert 0 (10 bytes)
        // and layer 1 expert 0 (20); step 1 layer 1 expert 1 (20); step 2 layer 0 expert 1 (10)
        // and layer 1 expert 2 (20); step 3 layer 0 expert 0 again. All reads before step 3 are
        // new to the page cache, and step 3 re-reads layer 0 expert 0, which the page cache
        // holds: 10 page-cache bytes, no SSD bytes (layer 1 hits).
        let outcome = replay(1, 1000);
        let bytes: Vec<(u64, u64)> = outcome
            .decode
            .iter()
            .map(|s| (s.page_cache_bytes, s.ssd_bytes))
            .collect();
        assert_eq!(bytes, vec![(0, 20), (0, 30), (10, 0)]);
    }

    /// The gain of `slots` slots for layer `layer`: `GAINS[layer][slots]`.
    const GAINS: [[u64; 4]; 3] = [[30, 3, 6, 60], [0, 9, 3, 0], [20, 90, 2, 2]];

    #[test]
    fn the_greedy_split_follows_the_gain_per_byte() {
        // Layers cost 10, 30, and 30 bytes per slot. Minimum 1 slot, maximum 3, budget 130. The
        // minimum costs 10 + 30 + 30 = 70 bytes, so 60 bytes are left. `gain(l, s)` is
        // GAINS[l][s]: layer 0 has 3 at s=1 and 6 at s=2; layer 1 has 9 at s=1 and 3 at s=2;
        // layer 2 has 90 at s=1 and 2 at s=2.
        // Round 1 (left 60, slots 1,1,1): per byte 3/10 = 0.3, 9/30 = 0.3, 90/30 = 3.0. Layer 2
        //   wins; left 30; slots 1,1,2.
        // Round 2 (left 30): 3/10 = 0.3, 9/30 = 0.3, 2/30 = 0.067. Layers 0 and 1 tie and the
        //   lower layer wins; left 20; slots 2,1,2.
        // Round 3 (left 20): layer 0 has 6/10 = 0.6; layers 1 and 2 cost 30 bytes and do not fit.
        //   Left 10; slots 3,1,2.
        // Round 4: layer 0 is at the maximum, and the other layers do not fit in 10 bytes.
        let split = greedy_split(&[10, 30, 30], 130, 1, 3, |layer, s| GAINS[layer][s]);
        assert_eq!(split, vec![3, 1, 2]);
    }

    #[test]
    fn a_budget_below_the_minimum_leaves_every_layer_at_the_minimum() {
        // The minimum of 2 slots costs 2 * (10 + 30) = 80 bytes, more than 50: nothing is left.
        let split = greedy_split(&[10, 30], 50, 2, 4, |_, _| 100);
        assert_eq!(split, vec![2, 2]);
    }

    #[test]
    fn the_minimum_slots_are_charged_before_the_greedy_rounds() {
        // Budget 100 over one layer of 10 bytes with minimum 4: 40 bytes are charged, 60 are
        // left, so 6 more slots fit but the maximum 8 stops it at 4 more.
        let split = greedy_split(&[10], 100, 4, 8, |_, _| 1);
        assert_eq!(split, vec![8]);
        // With maximum 20 the 60 left bytes give exactly 6 slots: 4 + 6 = 10.
        let split = greedy_split(&[10], 100, 4, 20, |_, _| 1);
        assert_eq!(split, vec![10]);
    }
}
