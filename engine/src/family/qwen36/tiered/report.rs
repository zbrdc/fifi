//! The per-token counters of the tiered decode (OFF-2 C1).
//!
//! One [`TokenReport`] aggregates what a decode token cost in the tier split: which
//! routed experts their layer bank served, which the CPU served, how many serves fell
//! through the page cache to the SSD, and the swaps the token boundary decided. The
//! report is plain data for the caller's accounting; the tiered host fills it (OFF-2 D4) and the
//! gate prints it (OFF-2 D5c).

/// Counters of one decode token in the tier split.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenReport {
    /// Routed experts served from the VRAM bank of their layer.
    pub bank_hits: u32,
    /// Routed experts served on the CPU because their layer bank missed.
    pub cpu_served: u32,
    /// CPU serves whose expert bytes were not in the OS page cache (SSD loads).
    pub ssd_misses: u32,
    /// Bank changes executed at the token boundary: warm-up fills plus swaps.
    pub swaps: u32,
    /// Expert bytes read from the SSD for the misses above.
    pub ssd_bytes: u64,
    /// Router weight of every routed pick, in millionths (a weight of 0.25 adds 250,000): the weight mass the layer
    /// outputs are built from. WHY: the CPU share of PICKS says little when the popular experts are the missed ones.
    pub routed_weight_ppm: u64,
    /// The part of `routed_weight_ppm` carried by the picks served on the CPU.
    pub cpu_weight_ppm: u64,
}

impl TokenReport {
    /// Adds every counter of `other` into `self`, so a window sums its tokens. WHY:
    /// the engine merges one report per layer and per step into the window total.
    pub fn merge(&mut self, other: &TokenReport) {
        let TokenReport {
            bank_hits,
            cpu_served,
            ssd_misses,
            swaps,
            ssd_bytes,
            routed_weight_ppm,
            cpu_weight_ppm,
        } = *other;
        self.bank_hits += bank_hits;
        self.cpu_served += cpu_served;
        self.ssd_misses += ssd_misses;
        self.swaps += swaps;
        self.ssd_bytes += ssd_bytes;
        self.routed_weight_ppm += routed_weight_ppm;
        self.cpu_weight_ppm += cpu_weight_ppm;
    }
}

#[cfg(test)]
mod tests {
    use super::TokenReport;

    #[test]
    fn token_report_merge_adds_every_counter() {
        let mut left = TokenReport {
            bank_hits: 3,
            cpu_served: 5,
            ssd_misses: 2,
            swaps: 1,
            ssd_bytes: 1_000,
            routed_weight_ppm: 900_000,
            cpu_weight_ppm: 400_000,
        };
        left.merge(&TokenReport {
            bank_hits: 4,
            cpu_served: 7,
            ssd_misses: 9,
            swaps: 6,
            ssd_bytes: 2_048,
            routed_weight_ppm: 100_000,
            cpu_weight_ppm: 100_000,
        });
        assert_eq!(
            left,
            TokenReport {
                bank_hits: 7,
                cpu_served: 12,
                ssd_misses: 11,
                swaps: 7,
                ssd_bytes: 3_048,
                routed_weight_ppm: 1_000_000,
                cpu_weight_ppm: 500_000,
            }
        );
        let mut empty = TokenReport::default();
        empty.merge(&TokenReport::default());
        assert_eq!(empty, TokenReport::default());
    }
}
