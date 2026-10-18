//! One layer's expert cache under a replacement policy (S3 plan section 3.5).
//!
//! A step selects `top_k` distinct experts. Hits are the selected experts that the cache holds
//! before the step. Every miss must enter the cache, because the device reads experts only from
//! cache slots; the victims are never experts of the same step. The victim is the evictable
//! resident with the smallest key, ties to the lower expert id:
//!
//! - LRU: the key is the step of the last use.
//! - LFU: the key is the use count since the start, kept for evicted experts too. With a half-life
//!   `H`, every count halves (integer division) after each `H` admitted steps.
//! - Hot: the key is the decayed hotness. Each use of an expert adds 1, and every hotness halves
//!   after each `H` admitted steps (the half-life), as an exponential decay of `2^(-1/H)` per
//!   step. Ties go to the less recently used expert. Evicted experts keep their hotness.
//! - Belady: the key falls with the distance to the next use, so the expert used furthest in the
//!   future, or never again, goes first. It is the bound, not a candidate.
//! - Static: a fixed core of the most frequent experts of the training prompts never leaves. The
//!   other slots form an LRU tail that takes the misses.
//!
//! With `admit` false (prompt bypass) a step counts its hits and changes nothing.

use std::collections::BTreeSet;

/// The replacement rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Lru,
    Lfu,
    Belady,
    Static,
    /// Decayed hotness with an LRU tie-break.
    Hot,
    /// One cache shared by all layers with least-stale eviction. It runs in `PoolCache`.
    Pool,
    /// The bound of `Pool`: one shared cache with furthest-next-use eviction.
    PoolBelady,
    /// No resident cache: the macOS page cache, an LRU over expert byte ranges, serves every read.
    /// It runs in `simulate`.
    PageOnly,
}

impl Kind {
    /// Whether the policy is a candidate for the per-layer design. Belady, the pool policies, and
    /// the page-cache baseline are bounds, another design, or no cache, so they stay out of the
    /// decision value `h*`.
    pub fn is_candidate(self) -> bool {
        !matches!(
            self,
            Kind::Belady | Kind::Pool | Kind::PoolBelady | Kind::PageOnly
        )
    }

    /// The bound of this policy: the pooled Belady for the pool and the page-cache baseline
    /// (both share their items over the layers), else the per-layer Belady.
    pub fn bound(self) -> Kind {
        if matches!(self, Kind::Pool | Kind::PageOnly) {
            Kind::PoolBelady
        } else {
            Kind::Belady
        }
    }
}

/// The size of a static core for `slots` slots and `top_k` experts per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Core {
    /// No core.
    None,
    /// `slots - top_k` core experts and a tail of `top_k` slots for the misses of one step.
    AllButStep,
    /// `slots / 2` core experts and an LRU tail of the rest.
    Half,
}

/// A named policy of the replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub name: &'static str,
    pub kind: Kind,
    /// LFU: halve every count after this many admitted steps. Hot: the half-life of the hotness
    /// in admitted steps.
    pub half_life: Option<u64>,
    pub core: Core,
}

/// The policies of S3 plan section 3.5, items 1 to 6.
pub const POLICIES: [Policy; 14] = [
    Policy::plain("lru", Kind::Lru),
    Policy::plain("lfu", Kind::Lfu),
    Policy::lfu_decay("lfu-h64", 64),
    Policy::lfu_decay("lfu-h256", 256),
    Policy::lfu_decay("lfu-h1024", 1024),
    Policy::hot("hot-h16", 16),
    Policy::hot("hot-h64", 64),
    Policy::hot("hot-h256", 256),
    Policy {
        name: "static",
        kind: Kind::Static,
        half_life: None,
        core: Core::AllButStep,
    },
    Policy {
        name: "static-half",
        kind: Kind::Static,
        half_life: None,
        core: Core::Half,
    },
    Policy::plain("belady", Kind::Belady),
    Policy::plain("pool", Kind::Pool),
    Policy::plain("belady-pool", Kind::PoolBelady),
    Policy::plain("page-only", Kind::PageOnly),
];

impl Policy {
    const fn plain(name: &'static str, kind: Kind) -> Self {
        Self {
            name,
            kind,
            half_life: None,
            core: Core::None,
        }
    }

    const fn lfu_decay(name: &'static str, half_life: u64) -> Self {
        Self {
            name,
            kind: Kind::Lfu,
            half_life: Some(half_life),
            core: Core::None,
        }
    }

    const fn hot(name: &'static str, half_life: u64) -> Self {
        Self {
            name,
            kind: Kind::Hot,
            half_life: Some(half_life),
            core: Core::None,
        }
    }

    /// The policy named `name`.
    pub fn named(name: &str) -> Option<Self> {
        POLICIES.iter().copied().find(|p| p.name == name)
    }

    /// The core size for `slots` slots.
    pub fn core_size(&self, slots: usize, top_k: usize) -> usize {
        match self.core {
            Core::None => 0,
            Core::AllButStep => slots.saturating_sub(top_k),
            Core::Half => slots / 2,
        }
    }
}

/// Never used again.
pub const NEVER: u32 = u32::MAX;

/// One layer's cache.
#[derive(Debug, Clone)]
pub struct LayerCache {
    kind: Kind,
    half_life: Option<u64>,
    /// Slots outside the core.
    tail: usize,
    resident: Vec<bool>,
    core: Vec<bool>,
    key: Vec<u64>,
    /// The second sort key: the step of the last use (Hot), else 0.
    tie: Vec<u64>,
    /// Evictable residents by `(key, tie, id)`.
    order: BTreeSet<(u64, u64, u16)>,
    counts: Vec<u64>,
    /// Hot: the hotness of every expert, scaled by `2^(clock / H)` so that it only grows.
    heat: Vec<f64>,
    /// Admitted steps so far.
    clock: u64,
    marked: Vec<bool>,
    /// LRU keys of real steps start above every warm key.
    base: u64,
}

impl LayerCache {
    /// A cache of `slots` slots over `experts` experts. `ranked` lists experts from the most to the
    /// least frequent in the training prompts. The static core takes its first experts. With
    /// `warm`, the remaining slots take the next experts, the more frequent as the more recent
    /// (LRU), with count 0 (LFU), or by their first use (Belady, from `first_use`).
    pub fn new(
        policy: Policy,
        experts: usize,
        slots: usize,
        top_k: usize,
        ranked: &[u16],
        warm: bool,
        first_use: &[u32],
    ) -> Result<Self, String> {
        if matches!(policy.kind, Kind::Pool | Kind::PoolBelady | Kind::PageOnly) {
            return Err(format!(
                "{}: this policy runs outside LayerCache",
                policy.name
            ));
        }
        let core_size = policy.core_size(slots, top_k);
        let tail = slots - core_size.min(slots);
        if tail < top_k || slots > experts {
            return Err(format!(
                "{}: {slots} slots with a core of {core_size} leave fewer than {top_k} tail slots or exceed {experts} experts",
                policy.name
            ));
        }
        if ranked.len() < slots || first_use.len() != experts {
            return Err("the ranking or the first uses do not cover the slots".into());
        }
        let mut cache = Self {
            kind: policy.kind,
            half_life: policy.half_life,
            tail,
            resident: vec![false; experts],
            core: vec![false; experts],
            key: vec![0; experts],
            tie: vec![0; experts],
            order: BTreeSet::new(),
            counts: vec![0; experts],
            heat: vec![0.0; experts],
            clock: 0,
            marked: vec![false; experts],
            base: experts as u64 + 1,
        };
        for &e in &ranked[..core_size] {
            cache.resident[usize::from(e)] = true;
            cache.core[usize::from(e)] = true;
        }
        if warm {
            for (rank, &e) in ranked[core_size..slots].iter().enumerate() {
                let key = match cache.kind {
                    Kind::Lru | Kind::Static | Kind::Pool | Kind::PageOnly => {
                        ((slots - rank) as u64, 0)
                    }
                    Kind::Lfu => (0, 0),
                    Kind::Hot => (0, (slots - rank) as u64),
                    Kind::Belady | Kind::PoolBelady => (belady_key(first_use[usize::from(e)]), 0),
                };
                cache.insert(e, key);
            }
        }
        Ok(cache)
    }

    fn insert(&mut self, e: u16, (key, tie): (u64, u64)) {
        let i = usize::from(e);
        self.resident[i] = true;
        self.key[i] = key;
        self.tie[i] = tie;
        self.order.insert((key, tie, e));
    }

    fn remove(&mut self, e: u16) {
        let i = usize::from(e);
        self.resident[i] = false;
        self.order.remove(&(self.key[i], self.tie[i], e));
    }

    /// The sort keys of `e` at the current step. `next` is its next admitted use (Belady).
    fn key_of(&self, e: u16, next: u32) -> (u64, u64) {
        match self.kind {
            Kind::Lru | Kind::Static | Kind::Pool | Kind::PageOnly => (self.base + self.clock, 0),
            Kind::Lfu => (self.counts[usize::from(e)], 0),
            // The bits of a non-negative float sort like the float.
            Kind::Hot => (self.heat[usize::from(e)].to_bits(), self.base + self.clock),
            Kind::Belady | Kind::PoolBelady => (belady_key(next), 0),
        }
    }

    /// Runs one step on `sel` and returns its hits. `misses` receives the selected experts that the
    /// cache did not hold. `next[r]` is the next admitted step that selects `sel[r]`, or `NEVER`
    /// (Belady only; other policies ignore it).
    pub fn access(
        &mut self,
        sel: &[u16],
        admit: bool,
        next: &[u32],
        misses: &mut Vec<u16>,
    ) -> Result<u32, String> {
        misses.clear();
        let mut hits = 0;
        for &e in sel {
            if self.resident[usize::from(e)] {
                hits += 1;
            } else {
                misses.push(e);
            }
        }
        if !admit {
            return Ok(hits);
        }
        let weight = match (self.kind, self.half_life) {
            (Kind::Hot, Some(h)) => (self.clock as f64 / h as f64).exp2(),
            _ => 1.0,
        };
        if !weight.is_finite() {
            return Err("the hotness overflows: the half-life is too short for this run".into());
        }
        for &e in sel {
            self.marked[usize::from(e)] = true;
            self.counts[usize::from(e)] += 1;
            self.heat[usize::from(e)] += weight;
        }
        for (rank, &e) in sel.iter().enumerate() {
            let i = usize::from(e);
            let next_use = next.get(rank).copied().unwrap_or(NEVER);
            if self.resident[i] && !self.core[i] {
                self.order.remove(&(self.key[i], self.tie[i], e));
                let (key, tie) = self.key_of(e, next_use);
                self.key[i] = key;
                self.tie[i] = tie;
                self.order.insert((key, tie, e));
            } else if !self.resident[i] {
                if self.order.len() >= self.tail {
                    let victim = self
                        .order
                        .iter()
                        .map(|&(_, _, v)| v)
                        .find(|&v| !self.marked[usize::from(v)])
                        .ok_or("no victim outside the step")?;
                    self.remove(victim);
                }
                let key = self.key_of(e, next_use);
                self.insert(e, key);
            }
        }
        for &e in sel {
            self.marked[usize::from(e)] = false;
        }
        self.clock += 1;
        if let Some(h) = self.half_life
            && self.kind == Kind::Lfu
            && self.clock.is_multiple_of(h)
        {
            for count in &mut self.counts {
                *count /= 2;
            }
            let evictable: Vec<u16> = self.order.iter().map(|&(_, _, e)| e).collect();
            self.order.clear();
            for e in evictable {
                let key = self.counts[usize::from(e)];
                self.key[usize::from(e)] = key;
                self.order.insert((key, 0, e));
            }
        }
        Ok(hits)
    }
}

/// Larger keys for nearer uses, so the smallest key is the furthest next use.
fn belady_key(next: u32) -> u64 {
    u64::from(u32::MAX - next)
}
