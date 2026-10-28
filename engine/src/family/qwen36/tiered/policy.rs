//! The tier policy of the per-layer expert banks (OFF-2 C1, packet policy P).
//!
//! WHAT: decayed popularity scores per (layer, expert), the resident set of every
//! layer bank, and a swap credit that buys a bounded number of swaps per token.
//! WHY: a 6 GiB GPU cannot hold 40 x 256 experts, so the engine keeps the hot few
//! of each layer on the GPU and pays a fixed swap budget for the rest. The
//! semantics mirror the python model `tier_adaptive.adaptive_per_layer` whose
//! numbers the off-t trace replay checks. Deterministic: no clock, no randomness,
//! no threads. Driven by `TieredHost` (OFF-2 D4/D5).

use super::{PolicyMode, TieredConfig};

/// The experts a [`PolicyMode::Static`] policy holds in every layer bank: at most
/// `bank_slots_per_layer` ids per layer, ranked by a popularity profile.
pub struct StaticProfile {
    /// The held expert ids of each layer, ascending.
    held: Vec<Vec<u16>>,
}

impl StaticProfile {
    /// Picks per layer the `bank_slots_per_layer` experts with the highest counts.
    /// WHY: a static bank holds the experts the profile calls hottest so the GPU
    /// never waits on a fill. Ties go to the lowest expert id, so equal counts
    /// always build the same profile.
    pub fn from_counts(cfg: &TieredConfig, counts: &[Vec<u64>]) -> Result<Self, String> {
        if counts.len() != cfg.layers {
            return Err(format!(
                "static profile: {} rows for {} layers",
                counts.len(),
                cfg.layers
            ));
        }
        if cfg.experts > u16::MAX as usize + 1 {
            return Err(format!(
                "static profile: {} experts do not fit u16 ids",
                cfg.experts
            ));
        }
        let mut held = Vec::with_capacity(cfg.layers);
        for (layer, row) in counts.iter().enumerate() {
            if row.len() != cfg.experts {
                return Err(format!(
                    "static profile: layer {layer} has {} counts",
                    row.len()
                ));
            }
            let mut ids: Vec<u16> = (0..cfg.experts as u16).collect();
            ids.sort_by(|&a, &b| row[b as usize].cmp(&row[a as usize]).then(a.cmp(&b)));
            ids.truncate(cfg.bank_slots_per_layer);
            ids.sort_unstable();
            held.push(ids);
        }
        Ok(Self { held })
    }

    /// The held experts of `layer`, ascending; empty for an unknown layer.
    pub fn held(&self, layer: usize) -> &[u16] {
        self.held.get(layer).map_or(&[], Vec::as_slice)
    }
}

/// One bank change decided at a token boundary: `promote` enters the bank of
/// `layer`; `evict` is the expert it replaces, or `None` while the bank still has
/// a free slot (a warm-up fill).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swap {
    /// The layer whose bank changes.
    pub layer: u16,
    /// The expert entering the bank.
    pub promote: u16,
    /// The expert leaving the bank, or `None` for a fill into a free slot.
    pub evict: Option<u16>,
}

/// The tier policy: decayed popularity scores, the per-layer banks and the swap
/// credit. `observe` scores one token's routes and reports its bank hits;
/// `boundary` decides the swaps of the token boundary. WHY: the bank must follow
/// the routing popularity without thrashing and without an unbounded SSD burst.
pub struct ExpertPolicy {
    cfg: TieredConfig,
    growth: f64,
    /// The current score increment: it grows by `growth` (`1 / decay`) every token
    /// and drags every score into a renormalisation once it passes `renorm_at`,
    /// so the f64 scores can never overflow.
    scale: f64,
    /// The renormalisation threshold: 2^64 in production, smaller in tests.
    renorm_at: f64,
    /// Decayed popularity scores, `[layer][expert]`, always >= 0.
    scores: Vec<Vec<f64>>,
    /// Bank membership, `[layer][expert]`; `bank_len` counts its row sums.
    resident: Vec<Vec<bool>>,
    bank_len: Vec<usize>,
    /// The experts routed by the current token that their bank missed, in token
    /// order (layer order, then route order), duplicate-free.
    touched: Vec<(u16, u16)>,
    /// Token stamp per `[layer][expert]` that keeps `touched` duplicate-free.
    stamps: Vec<Vec<u64>>,
    token: u64,
    /// Swap credit: grows by `swaps_per_token` per token; its integer part is the
    /// boundary budget and the fraction carries over.
    credit: f64,
}

impl ExpertPolicy {
    /// Builds the policy for a checked `cfg`. A `Static` policy REQUIRES a profile
    /// and starts with every bank full of it; an `Adaptive` policy takes no profile
    /// and starts with empty banks that the warm-up fills with the first routed
    /// experts. `Err` reports a broken config or a mismatched profile.
    pub fn new(cfg: &TieredConfig, profile: Option<&StaticProfile>) -> Result<Self, String> {
        cfg.check()?;
        if cfg.experts > u16::MAX as usize + 1 {
            return Err(format!(
                "tier policy: {} experts do not fit u16 ids",
                cfg.experts
            ));
        }
        let mut policy = Self {
            cfg: cfg.clone(),
            growth: 1.0 / cfg.decay,
            scale: 1.0,
            renorm_at: 2f64.powi(64),
            scores: vec![vec![0.0; cfg.experts]; cfg.layers],
            resident: vec![vec![false; cfg.experts]; cfg.layers],
            bank_len: vec![0; cfg.layers],
            touched: Vec::new(),
            stamps: vec![vec![0; cfg.experts]; cfg.layers],
            token: 0,
            credit: 0.0,
        };
        let profile = match (cfg.mode, profile) {
            (PolicyMode::Adaptive, None) => return Ok(policy),
            (PolicyMode::Adaptive, Some(_)) => {
                return Err(String::from(
                    "tier policy: an adaptive policy starts empty, pass no profile",
                ));
            }
            (PolicyMode::Static, None) => {
                return Err(String::from(
                    "tier policy: a static policy requires a profile",
                ));
            }
            (PolicyMode::Static, Some(profile)) => profile,
        };
        let fits = profile.held.len() == cfg.layers
            && profile
                .held
                .iter()
                .all(|h| h.len() <= cfg.bank_slots_per_layer);
        if !fits {
            return Err(String::from(
                "tier policy: the profile does not fit the config banks",
            ));
        }
        for (layer, held) in profile.held.iter().enumerate() {
            policy.bank_len[layer] = held.len();
            for &expert in held {
                policy.resident[layer][expert as usize] = true;
            }
        }
        Ok(policy)
    }

    /// Whether `expert` is in the bank of `layer`. `false` for an unknown layer or
    /// expert, so a stale id can never panic a caller.
    pub fn is_resident(&self, layer: usize, expert: usize) -> bool {
        self.resident
            .get(layer)
            .and_then(|bank| bank.get(expert))
            .copied()
            .unwrap_or(false)
    }

    /// Marks `expert` resident in the bank of `layer` before the first token (a preload): the
    /// caller then fills the slot through the residency table. Only an adaptive policy takes it,
    /// only for an expert that is not resident, and only while the bank has a free slot.
    pub fn preload(&mut self, layer: usize, expert: u16) -> Result<(), String> {
        if self.cfg.mode != PolicyMode::Adaptive {
            return Err(String::from(
                "tier policy: only an adaptive policy takes a preload",
            ));
        }
        let (l, e) = (layer, usize::from(expert));
        if l >= self.cfg.layers || e >= self.cfg.experts {
            return Err(format!(
                "tier policy: preload of expert {e} in layer {l} is out of range"
            ));
        }
        if self.resident[l][e] || self.bank_len[l] >= self.cfg.bank_slots_per_layer {
            return Err(format!(
                "tier policy: expert {e} of layer {l} is resident or the bank is full"
            ));
        }
        self.resident[l][e] = true;
        self.bank_len[l] += 1;
        Ok(())
    }

    /// Scores one token: `routes[layer]` holds the top-k expert ids of that layer,
    /// in routing order. Every routed expert gains the current increment, and the
    /// routed experts their bank missed become the token's touches. Returns, per
    /// layer and in the same order, which routed experts are bank hits.
    pub fn observe(&mut self, routes: &[Vec<u16>]) -> Vec<Vec<bool>> {
        assert_eq!(routes.len(), self.cfg.layers, "one route list per layer");
        self.token += 1;
        self.touched.clear();
        self.scale *= self.growth;
        if self.scale > self.renorm_at {
            self.renormalise();
        }
        let mut hits = Vec::with_capacity(routes.len());
        for (layer, route) in routes.iter().enumerate() {
            assert!(route.len() <= self.cfg.top_k, "layer {layer} exceeds top_k");
            let mut row = Vec::with_capacity(route.len());
            for &expert in route {
                let expert = expert as usize;
                assert!(expert < self.cfg.experts, "expert id out of range");
                self.scores[layer][expert] += self.scale;
                let hit = self.resident[layer][expert];
                if !hit && self.stamps[layer][expert] != self.token {
                    self.stamps[layer][expert] = self.token;
                    self.touched.push((layer as u16, expert as u16));
                }
                row.push(hit);
            }
            hits.push(row);
        }
        hits
    }

    /// Decides the swaps of the token boundary and applies them to the internal
    /// banks; the caller mirrors them through the residency table. Warm-up fills
    /// come first, in token order, while a layer bank still has a free slot; then
    /// the accumulated credit buys full-bank swaps, best score ratio first. A
    /// static policy returns an empty list forever; the touches of the token are
    /// consumed, so a second call without `observe` returns nothing.
    pub fn boundary(&mut self) -> Vec<Swap> {
        let touched = std::mem::take(&mut self.touched);
        if self.cfg.mode != PolicyMode::Adaptive {
            return Vec::new();
        }
        let slots = self.cfg.bank_slots_per_layer;
        let mut swaps = Vec::new();
        for &(layer, expert) in &touched {
            let (l, e) = (layer as usize, expert as usize);
            if self.bank_len[l] < slots && !self.resident[l][e] {
                self.resident[l][e] = true;
                self.bank_len[l] += 1;
                swaps.push(Swap {
                    layer,
                    promote: expert,
                    evict: None,
                });
            }
        }
        self.credit += self.cfg.swaps_per_token;
        let spent = self.credit.floor();
        self.credit -= spent;
        let budget = spent as usize;
        if budget > 0 {
            // Candidates: touched experts whose layer bank is full. The victim is
            // the lowest-score resident of the same bank; the swap only pays if
            // the candidate outscores it.
            let mut options = Vec::new();
            for &(layer, expert) in &touched {
                let (l, e) = (layer as usize, expert as usize);
                if self.resident[l][e] || self.bank_len[l] < slots {
                    continue;
                }
                let victim = self.lowest_of(l);
                if self.scores[l][e] > self.scores[l][victim] {
                    let ratio = self.scores[l][e] / self.scores[l][victim];
                    options.push((ratio, layer, expert, victim as u16));
                }
            }
            // Best ratio first; ties: lower layer, then lower expert id.
            options.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
            for &(_, layer, promote, victim) in options.iter().take(budget) {
                let l = layer as usize;
                // Two candidates of one layer can share a recorded victim; the
                // second swap then takes the current lowest resident instead.
                let evict = if self.resident[l][victim as usize] {
                    victim
                } else {
                    self.lowest_of(l) as u16
                };
                self.resident[l][evict as usize] = false;
                self.resident[l][promote as usize] = true;
                swaps.push(Swap {
                    layer,
                    promote,
                    evict: Some(evict),
                });
            }
        }
        swaps
    }

    /// The resident expert of `layer` with the lowest score; equal scores keep the
    /// lowest expert id because the scan is in id order. The bank must not be
    /// empty, which callers guarantee by checking it is full first.
    fn lowest_of(&self, layer: usize) -> usize {
        let scores = &self.scores[layer];
        let (mut best, mut best_score) = (usize::MAX, f64::INFINITY);
        for (expert, &resident) in self.resident[layer].iter().enumerate() {
            if resident && scores[expert] < best_score {
                best = expert;
                best_score = scores[expert];
            }
        }
        best
    }

    /// Multiplies every score and `scale` by 2^-64, an exact power-of-two step in
    /// f64, so the growing increment can never overflow. Ratios and comparisons
    /// stay put, so no swap decision moves.
    fn renormalise(&mut self) {
        let down = 2f64.powi(-64);
        for layer in &mut self.scores {
            for score in layer.iter_mut() {
                *score *= down;
            }
        }
        self.scale *= down;
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertPolicy, PolicyMode, StaticProfile, Swap, TieredConfig};
    use std::collections::{HashMap, VecDeque};

    /// The tiny deterministic LCG of the offline tests (rule 5): fixed seeds, no
    /// OS randomness.
    struct Lcg(u32);
    impl Lcg {
        fn below(&mut self, n: u32) -> u32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 8) % n
        }
    }

    /// A small adaptive config: 2 layers x 8 experts, top-2 routing, banks of
    /// `slots` experts, `rate` swaps per token, decay 0.999.
    fn adaptive(slots: usize, rate: f64) -> TieredConfig {
        TieredConfig {
            layers: 2,
            experts: 8,
            top_k: 2,
            bank_slots_per_layer: slots,
            mode: PolicyMode::Adaptive,
            swaps_per_token: rate,
            decay: 0.999,
        }
    }

    /// The expected shape of a warm-up fill into a free slot.
    fn fill(layer: u16, expert: u16) -> Swap {
        Swap {
            layer,
            promote: expert,
            evict: None,
        }
    }

    /// One random token for `cfg`: per layer `top_k` ids drawn from `Lcg`.
    fn routes(rng: &mut Lcg, cfg: &TieredConfig) -> Vec<Vec<u16>> {
        (0..cfg.layers)
            .map(|_| {
                (0..cfg.top_k)
                    .map(|_| rng.below(cfg.experts as u32) as u16)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn policy_never_swaps_more_than_the_credit() {
        let cfg = adaptive(2, 0.25);
        let mut rng = Lcg(7);
        let mut policy = ExpertPolicy::new(&cfg, None).unwrap();
        let mut swaps = 0usize;
        for _ in 0..4_000 {
            let token = routes(&mut rng, &cfg);
            policy.observe(&token);
            swaps += policy.boundary().len();
        }
        // floor(0.25 * 4_000) credit swaps plus at most layers * slots warm-up fills.
        let cap = 1_000 + cfg.layers * cfg.bank_slots_per_layer;
        assert!(swaps <= cap, "{swaps} swaps exceed the credit cap {cap}");
    }

    #[test]
    fn a_victim_is_the_lowest_score_of_its_own_layer() {
        let cfg = adaptive(2, 1.0);
        let mut policy = ExpertPolicy::new(&cfg, None).unwrap();
        // Token 1 warms both banks up: layer 0 with {0, 1}, layer 1 with {4, 5}.
        policy.observe(&[vec![0, 1], vec![4, 5]]);
        assert_eq!(
            policy.boundary(),
            vec![fill(0, 0), fill(0, 1), fill(1, 4), fill(1, 5)]
        );
        // Token 2 routes 2 (layer 0) and 6 (layer 1). Both candidates beat only the
        // coldest resident of their OWN layer (0 and 4); the ratios tie, so the
        // lower layer takes the budget of one swap.
        policy.observe(&[vec![2, 1], vec![6, 5]]);
        assert_eq!(
            policy.boundary(),
            vec![Swap {
                layer: 0,
                promote: 2,
                evict: Some(0)
            }]
        );
        assert!(policy.is_resident(0, 2) && !policy.is_resident(0, 0));
        assert!(
            policy.is_resident(1, 4) && policy.is_resident(1, 5),
            "layer 1 untouched"
        );
        // Token 3 repeats the route: layer 0 now holds the warm pair, so the swap
        // evicts the coldest resident of layer 1 instead.
        policy.observe(&[vec![2, 1], vec![6, 5]]);
        assert_eq!(
            policy.boundary(),
            vec![Swap {
                layer: 1,
                promote: 6,
                evict: Some(4)
            }]
        );
    }

    #[test]
    fn warm_up_fills_each_bank_with_the_first_routed_experts() {
        let cfg = adaptive(3, 0.25);
        let mut policy = ExpertPolicy::new(&cfg, None).unwrap();
        policy.observe(&[vec![3, 1], vec![6, 2]]);
        assert_eq!(
            policy.boundary(),
            vec![fill(0, 3), fill(0, 1), fill(1, 6), fill(1, 2)]
        );
        policy.observe(&[vec![5, 3], vec![6, 0]]);
        // 3 and 6 are already resident; 5 and 0 take the last free slots.
        assert_eq!(policy.boundary(), vec![fill(0, 5), fill(1, 0)]);
        assert!(policy.is_resident(0, 1) && policy.is_resident(0, 3) && policy.is_resident(0, 5));
        assert!(policy.is_resident(1, 0) && policy.is_resident(1, 2) && policy.is_resident(1, 6));
    }

    #[test]
    fn static_profile_picks_the_top_experts_per_layer_with_lowest_id_ties() {
        let cfg = TieredConfig {
            mode: PolicyMode::Static,
            ..adaptive(3, 0.25)
        };
        let counts = [vec![5, 9, 9, 1, 9, 0, 0, 0], vec![7, 7, 7, 7, 2, 1, 0, 0]];
        let profile = StaticProfile::from_counts(&cfg, &counts).unwrap();
        assert_eq!(
            profile.held(0),
            [1, 2, 4],
            "count ties go to the lowest ids"
        );
        assert_eq!(
            profile.held(1),
            [0, 1, 2],
            "a four-way tie keeps the first three"
        );
        assert!(
            StaticProfile::from_counts(&cfg, &counts[..1]).is_err(),
            "layer count"
        );
        assert!(
            StaticProfile::from_counts(&cfg, &[vec![1, 2], vec![]]).is_err(),
            "expert count"
        );
    }

    #[test]
    fn static_mode_never_swaps_and_requires_a_profile() {
        let cfg = TieredConfig {
            mode: PolicyMode::Static,
            ..adaptive(2, 0.25)
        };
        assert!(
            ExpertPolicy::new(&cfg, None).is_err(),
            "static needs a profile"
        );
        let profile = StaticProfile::from_counts(
            &cfg,
            &[vec![9, 0, 7, 0, 0, 0, 0, 0], vec![0, 0, 0, 0, 9, 8, 0, 0]],
        )
        .unwrap();
        assert_eq!(profile.held(0), [0, 2]);
        assert_eq!(profile.held(1), [4, 5]);
        assert!(
            ExpertPolicy::new(&adaptive(2, 0.25), Some(&profile)).is_err(),
            "adaptive takes no profile"
        );
        let mut policy = ExpertPolicy::new(&cfg, Some(&profile)).unwrap();
        let mut rng = Lcg(99);
        for _ in 0..50 {
            let token = routes(&mut rng, &cfg);
            policy.observe(&token);
            assert!(policy.boundary().is_empty(), "static never swaps");
        }
        assert!(policy.is_resident(0, 0) && policy.is_resident(0, 2));
        assert!(policy.is_resident(1, 4) && policy.is_resident(1, 5) && !policy.is_resident(0, 1));
    }

    #[test]
    fn scores_survive_renormalisation_without_changing_decisions() {
        let cfg = adaptive(3, 0.25);
        let mut plain = ExpertPolicy::new(&cfg, None).unwrap();
        let mut eager = ExpertPolicy::new(&cfg, None).unwrap();
        // 1.001^t passes 2.0 within the first 694 of the 2,000 tokens below, so the
        // eager run renormalises while the production threshold 2^64 never fires;
        // the exact 2^-64 step must leave every swap decision unchanged.
        eager.renorm_at = 2.0;
        let mut rng = Lcg(4242);
        let mut eager_swaps = 0usize;
        for _ in 0..2_000 {
            let token = routes(&mut rng, &cfg);
            plain.observe(&token);
            let plain_swaps = plain.boundary();
            eager.observe(&token);
            let eager_now = eager.boundary();
            assert_eq!(plain_swaps, eager_now);
            eager_swaps += eager_now.len();
        }
        assert!(eager_swaps > 0, "the stream must exercise real swaps");
    }

    #[test]
    #[ignore = "reads the mounted off-t route trace (OFFT_TRACE); run with: cargo test --lib tiered -- --ignored policy_replays_the_offt_trace_within_3_percent"]
    fn policy_replays_the_offt_trace_within_3_percent() {
        replay_off_t_trace(61, 5.02);
        replay_off_t_trace(42, 5.53);
    }

    /// Replays the whole off-t trace through the adaptive policy plus the page
    /// cache model of the python reference: an LRU of 7,350 experts refreshed by
    /// every non-bank access, left by every promoted expert, where a miss is an
    /// access in neither the bank nor the LRU. Misses count only on `g` steps and
    /// the per-step mean must sit within 3 percent of the reference number.
    fn replay_off_t_trace(bank: usize, expected: f64) {
        let Some(path) = std::env::var_os("OFFT_TRACE") else {
            eprintln!("skipping: OFFT_TRACE is not set (the off-t trace is not mounted)");
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        // One step per (prompt, mode, pos) group, 40 layer rows of 8 ids each.
        let mut steps: Vec<(bool, Vec<Vec<u16>>)> = Vec::new();
        let mut key = ("", "", "");
        for line in text.lines() {
            if line.starts_with('#') || line.starts_with("prompt\t") {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            let this = (fields[0], fields[1], fields[2]);
            if this != key {
                key = this;
                steps.push((fields[1] == "g", Vec::new()));
            }
            steps
                .last_mut()
                .unwrap()
                .1
                .push(fields[4..12].iter().map(|id| id.parse().unwrap()).collect());
        }
        let cfg = TieredConfig::qwen36(bank, PolicyMode::Adaptive);
        let mut policy = ExpertPolicy::new(&cfg, None).unwrap();
        // The LRU is a stamp per live expert plus a lazy queue: stale queue heads
        // are skipped, so the first live head is the least recently touched expert.
        let mut live: HashMap<(u16, u16), u64> = HashMap::new();
        let mut order: VecDeque<((u16, u16), u64)> = VecDeque::new();
        let (mut seq, mut misses, mut g_steps) = (0u64, 0u64, 0u64);
        for (is_g, rows) in &steps {
            let hits = policy.observe(rows);
            g_steps += *is_g as u64;
            for (layer, ids) in rows.iter().enumerate() {
                for (choice, &expert) in ids.iter().enumerate() {
                    if hits[layer][choice] {
                        continue;
                    }
                    let id = (layer as u16, expert);
                    seq += 1;
                    if !live.contains_key(&id) {
                        misses += *is_g as u64;
                    }
                    live.insert(id, seq);
                    order.push_back((id, seq));
                    while live.len() > 7_350 {
                        while let Some((old, stamp)) = order.pop_front() {
                            if live.get(&old) == Some(&stamp) {
                                live.remove(&old);
                                break;
                            }
                        }
                    }
                }
            }
            for swap in policy.boundary() {
                live.remove(&(swap.layer, swap.promote));
            }
        }
        let per_g_step = misses as f64 / g_steps as f64;
        let slack = expected * 0.03;
        assert!(
            (per_g_step - expected).abs() <= slack,
            "bank {bank}: {per_g_step:.3} SSD misses per g step, expected {expected} +- {slack:.3}"
        );
    }

    #[test]
    fn a_preload_marks_residents_until_the_bank_is_full_and_keeps_the_rules() {
        let cfg = TieredConfig {
            bank_slots_per_layer: 2,
            ..TieredConfig::qwen36(2, PolicyMode::Adaptive)
        };
        let mut policy = ExpertPolicy::new(&cfg, None).unwrap();
        assert!(policy.preload(0, 7).is_ok() && policy.preload(0, 9).is_ok());
        assert!(policy.is_resident(0, 7) && policy.is_resident(0, 9) && !policy.is_resident(1, 7));
        assert!(
            policy.preload(0, 11).is_err(),
            "the bank of layer 0 is full"
        );
        assert!(policy.preload(0, 7).is_err(), "already resident");
        assert!(
            policy.preload(cfg.layers, 1).is_err()
                && policy.preload(1, cfg.experts as u16).is_err()
        );
        // The preloaded experts count as hits and are never touches.
        let mut routes = vec![Vec::new(); cfg.layers];
        routes[0] = vec![7, 9, 3];
        assert_eq!(policy.observe(&routes)[0], vec![true, true, false]);
        let stat = TieredConfig::qwen36(2, PolicyMode::Static);
        let profile =
            StaticProfile::from_counts(&stat, &vec![vec![1; stat.experts]; stat.layers]).unwrap();
        let mut fixed = ExpertPolicy::new(&stat, Some(&profile)).unwrap();
        assert!(
            fixed.preload(0, 1).is_err(),
            "a static policy refuses a preload"
        );
    }
}
