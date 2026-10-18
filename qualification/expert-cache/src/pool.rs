//! One cache shared by all layers (T0X, an addition to S3 plan section 3.5).
//!
//! The pool holds `(layer, expert)` items. Its capacity is the sum of the per-layer slot counts,
//! so a pool and a per-layer cache of the same slots use the same memory. A step is one layer of
//! one token. Hits are the selected experts that the pool holds before the step. Every miss
//! enters the pool, and a victim is never an expert of the same step. A token is one admitted
//! trace step over all layers.
//!
//! - `LeastStale`: the staleness of an item is the current token index minus the token index of
//!   its last use. A warm item counts as used at token -1. The victim has the largest staleness.
//!   Among items with that staleness, the victim is the first in the order of layers `L, L-1,
//!   ..., 0, 47, ..., L+1` (`L` is the layer in service, 47 is the last layer), because the layer
//!   that comes up last in the layer cycle is needed furthest ahead. Inside one layer the lower
//!   expert id goes first.
//! - `Belady`: the victim is the item with the furthest next use in time `token * layers +
//!   layer`, or never. It is the bound of the pool, not a candidate.

use crate::cache::NEVER;
use std::collections::BTreeSet;

/// The replacement rule of a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolRule {
    LeastStale,
    Belady,
}

/// The pool.
#[derive(Debug, Clone)]
pub struct PoolCache {
    rule: PoolRule,
    layers: usize,
    experts: usize,
    capacity: usize,
    resident: Vec<bool>,
    key: Vec<u64>,
    marked: Vec<bool>,
    /// Residents by `(key, layer, u16::MAX - expert)`. The key is the last-use token plus 1
    /// (`LeastStale`, 0 for a warm item) or `u64::MAX` minus the next-use time (`Belady`).
    order: BTreeSet<(u64, u16, u16)>,
}

impl PoolCache {
    /// A pool over `slots.len()` layers of `experts` experts. The capacity is the sum of `slots`.
    /// With `warm`, the pool starts with `ranked[layer][..slots[layer]]` for every layer.
    /// `first[layer][expert]` is the first admitted step that selects the expert, or `NEVER`
    /// (`Belady` only).
    pub fn new(
        rule: PoolRule,
        experts: usize,
        slots: &[usize],
        top_k: usize,
        ranked: &[Vec<u16>],
        warm: bool,
        first: &[Vec<u32>],
    ) -> Result<Self, String> {
        let layers = slots.len();
        let capacity: usize = slots.iter().sum();
        if layers == 0 || layers >= usize::from(u16::MAX) || capacity < top_k {
            return Err(format!(
                "pool of {capacity} slots over {layers} layers cannot hold {top_k} experts"
            ));
        }
        if slots.iter().any(|&s| s > experts) || ranked.len() != layers || first.len() != layers {
            return Err("the slots, ranking, or first uses do not fit the layers".into());
        }
        let mut pool = Self {
            rule,
            layers,
            experts,
            capacity,
            resident: vec![false; layers * experts],
            key: vec![0; layers * experts],
            marked: vec![false; layers * experts],
            order: BTreeSet::new(),
        };
        if warm {
            for (layer, &count) in slots.iter().enumerate() {
                let (Some(list), uses) = (ranked[layer].get(..count), &first[layer]) else {
                    return Err("the ranking does not cover the slots".into());
                };
                if uses.len() != experts {
                    return Err("the first uses do not cover the experts".into());
                }
                for &e in list {
                    let key = pool.key_of(0, layer, uses[usize::from(e)]);
                    pool.insert(layer, e, key);
                }
            }
        }
        Ok(pool)
    }

    fn item(&self, layer: usize, e: u16) -> usize {
        layer * self.experts + usize::from(e)
    }

    fn insert(&mut self, layer: usize, e: u16, key: u64) {
        let i = self.item(layer, e);
        self.resident[i] = true;
        self.key[i] = key;
        self.order.insert((key, layer as u16, u16::MAX - e));
    }

    fn remove(&mut self, layer: usize, e: u16) {
        let i = self.item(layer, e);
        self.resident[i] = false;
        self.order
            .remove(&(self.key[i], layer as u16, u16::MAX - e));
    }

    /// The key of an item used now. `current` is the token index plus 1 (0 for a warm item), and
    /// `next` is the next admitted step that selects the item (`Belady` only).
    fn key_of(&self, current: u64, layer: usize, next: u32) -> u64 {
        match self.rule {
            PoolRule::LeastStale => current,
            PoolRule::Belady => {
                let time = if next == NEVER {
                    u64::MAX
                } else {
                    u64::from(next) * self.layers as u64 + layer as u64
                };
                u64::MAX - time
            }
        }
    }

    /// The victim while `layer` is in service: an item outside the step.
    fn victim(&self, layer: usize) -> Option<(usize, u16)> {
        let free = |&&(_, l, inv): &&(u64, u16, u16)| {
            !self.marked[self.item(usize::from(l), u16::MAX - inv)]
        };
        let found = match self.rule {
            PoolRule::Belady => self.order.iter().find(free),
            PoolRule::LeastStale => {
                let &(oldest, _, _) = self.order.first()?;
                let cut = layer as u16 + 1;
                let served = self.order.range((oldest, 0, 0)..(oldest, cut, 0)).rev();
                let others = self
                    .order
                    .range((oldest, cut, 0)..=(oldest, u16::MAX, u16::MAX))
                    .rev();
                served.chain(others).find(free)
            }
        };
        found.map(|&(_, l, inv)| (usize::from(l), u16::MAX - inv))
    }

    /// Runs the step of `layer` at admitted token `token` and returns its hits. `misses` receives
    /// the selected experts that the pool did not hold. `next[r]` is the next admitted step that
    /// selects `sel[r]` in this layer, or `NEVER` (`Belady` only). With `admit` false, the step
    /// counts its hits and changes nothing.
    pub fn access(
        &mut self,
        token: u32,
        layer: usize,
        sel: &[u16],
        admit: bool,
        next: &[u32],
        misses: &mut Vec<u16>,
    ) -> Result<u32, String> {
        misses.clear();
        let mut hits = 0;
        for &e in sel {
            if self.resident[self.item(layer, e)] {
                hits += 1;
            } else {
                misses.push(e);
            }
        }
        if !admit {
            return Ok(hits);
        }
        let current = u64::from(token) + 1;
        for &e in sel {
            let i = self.item(layer, e);
            self.marked[i] = true;
        }
        // Refresh every hit first, so that the experts of this step are the newest items.
        for (rank, &e) in sel.iter().enumerate() {
            if self.resident[self.item(layer, e)] {
                let key = self.key_of(current, layer, next.get(rank).copied().unwrap_or(NEVER));
                self.remove(layer, e);
                self.insert(layer, e, key);
            }
        }
        for (rank, &e) in sel.iter().enumerate() {
            if self.resident[self.item(layer, e)] {
                continue;
            }
            if self.order.len() >= self.capacity {
                let (victim_layer, victim) =
                    self.victim(layer).ok_or("no victim outside the step")?;
                self.remove(victim_layer, victim);
            }
            let key = self.key_of(current, layer, next.get(rank).copied().unwrap_or(NEVER));
            self.insert(layer, e, key);
        }
        for &e in sel {
            let i = self.item(layer, e);
            self.marked[i] = false;
        }
        Ok(hits)
    }
}
