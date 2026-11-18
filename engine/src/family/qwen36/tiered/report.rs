//! The per-token counters of the tiered decode.
//!
//! One [`TokenReport`] aggregates what a decode token cost in the tier split: which
//! routed experts their layer bank served, which the CPU served, how many serves fell
//! through the page cache to the SSD, and the swaps the token boundary decided. The
//! report is plain data for the caller's accounting; the tiered host fills it and the
//! gate prints it.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenReport {
    pub bank_hits: u32,
    pub cpu_served: u32,
    /// Chunk-prefill picks served by the cold staging bank on the device (GPU-served
    /// misses: not bank hits, not CPU pool jobs). Zero on the per-token path.
    pub cold_served: u32,
    /// CPU serves whose expert bytes were not in the OS page cache (SSD loads).
    pub ssd_misses: u32,
    /// Bank changes executed at the token boundary: warm-up fills plus swaps.
    pub swaps: u32,
    pub ssd_bytes: u64,
    /// Router weight of every routed pick, in millionths (a weight of 0.25 adds 250,000): the weight mass the layer
    /// outputs are built from. WHY: the CPU share of PICKS says little when the popular experts are the missed ones.
    pub routed_weight_ppm: u64,
    pub cpu_weight_ppm: u64,
    /// Router weight of the cold-staged picks, in millionths (the staging share of mass).
    pub cold_weight_ppm: u64,
    /// Router-guess prefetch fills issued (advice promotions that reached the fill path).
    pub guess_fills: u32,
    /// Prefetch promotions that evicted a resident expert (the thrash-bound counter).
    pub guess_evicts: u32,
    /// Advice layers consumed against a real route (the RecallMeter sample count).
    pub guess_samples: u64,
    /// Advised experts that the consumed route actually picked (the RecallMeter hits).
    pub guess_hits: u64,
}

impl TokenReport {
    /// WHY: the engine merges one report per layer and per step into the window
    /// total.
    pub fn merge(&mut self, other: &TokenReport) {
        let TokenReport {
            bank_hits,
            cpu_served,
            cold_served,
            ssd_misses,
            swaps,
            ssd_bytes,
            routed_weight_ppm,
            cpu_weight_ppm,
            cold_weight_ppm,
            guess_fills,
            guess_evicts,
            guess_samples,
            guess_hits,
        } = *other;
        self.bank_hits += bank_hits;
        self.cpu_served += cpu_served;
        self.cold_served += cold_served;
        self.ssd_misses += ssd_misses;
        self.swaps += swaps;
        self.ssd_bytes += ssd_bytes;
        self.routed_weight_ppm += routed_weight_ppm;
        self.cpu_weight_ppm += cpu_weight_ppm;
        self.cold_weight_ppm += cold_weight_ppm;
        self.guess_fills += guess_fills;
        self.guess_evicts += guess_evicts;
        self.guess_samples += guess_samples;
        self.guess_hits += guess_hits;
    }
}

#[cfg(test)]
mod tests {
    use super::TokenReport;

    #[test]
    fn token_report_merge_adds_every_counter() {
        let mut left = TokenReport {
            cold_served: 1,
            cold_weight_ppm: 11,
            bank_hits: 3,
            cpu_served: 5,
            ssd_misses: 2,
            swaps: 1,
            ssd_bytes: 1_000,
            routed_weight_ppm: 900_000,
            cpu_weight_ppm: 400_000,
            guess_fills: 2,
            guess_evicts: 1,
            guess_samples: 6,
            guess_hits: 4,
        };
        left.merge(&TokenReport {
            cold_served: 2,
            cold_weight_ppm: 22,
            bank_hits: 4,
            cpu_served: 7,
            ssd_misses: 9,
            swaps: 6,
            ssd_bytes: 2_048,
            routed_weight_ppm: 100_000,
            cpu_weight_ppm: 100_000,
            guess_fills: 5,
            guess_evicts: 2,
            guess_samples: 9,
            guess_hits: 3,
        });
        assert_eq!(
            left,
            TokenReport {
                cold_served: 3,
                cold_weight_ppm: 33,
                bank_hits: 7,
                cpu_served: 12,
                ssd_misses: 11,
                swaps: 7,
                ssd_bytes: 3_048,
                routed_weight_ppm: 1_000_000,
                cpu_weight_ppm: 500_000,
                guess_fills: 7,
                guess_evicts: 3,
                guess_samples: 15,
                guess_hits: 7,
            }
        );
        let mut empty = TokenReport::default();
        empty.merge(&TokenReport::default());
        assert_eq!(empty, TokenReport::default());
    }
}
