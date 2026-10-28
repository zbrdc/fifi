//! The section-timing marks of one captured decode token (packet B1, in-graph section timer).
//! Pure host code: which token sections carry in-graph timing events, and how the marked
//! sections extrapolate to a whole token.
//!
//! The idea is DeepInfer's graph timer (ledger `deepinfer-774ce24/B-kernels-moe-attention-quant.md`
//! lines 40 and 71): events recorded with the external-record flag inside a stream capture
//! become event-record nodes of the captured graph, so a replay re-records them and the host
//! reads the sections after the replay's read-back, with no re-capture. Only one layer of each
//! run of identical layers is timed and scaled by the run size, which keeps the node count low
//! (every extra event node costs replay time on a ~6.6 ms step). DeepInfer marks the second
//! layer of each group; a run of one layer marks its only layer, since a singleton has no
//! second and is its own representative.
//!
//! The sections follow `step_ops::token_ops`: the embedding, one mixer+MoE pair per marked
//! layer, the tail (`FinalNorm`..`Advance`), and one whole-token pair around everything.

use crate::EngError;

/// One marked run of identical layer kinds: the run is timed only at `layer` and the reading
/// stands for `size` layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MarkedGroup {
    /// The layer whose mixer+MoE pair carries the timing pair (the second of the run, or the
    /// only layer of a singleton run).
    pub(super) layer: usize,
    /// Layers of the run, the marked one included.
    pub(super) size: usize,
}

/// The timing marks of one captured token: one pair per run of identical layer kinds.
#[derive(Debug, Clone)]
pub(super) struct SectionPlan {
    pub(super) groups: Vec<MarkedGroup>,
}

/// One section reading, reconciled: the marked milliseconds plus the extrapolation.
#[derive(Debug, Clone)]
pub(super) struct SectionTimes {
    embed_ms: f32,
    tail_ms: f32,
    group_ms: Vec<f32>,
    sizes: Vec<usize>,
    whole_ms: f32,
}

/// The marks of a layer-kind table: the second layer of each maximal run of equal kinds, in
/// layer order.
pub(super) fn section_plan(kinds: &[crate::family::AttentionKind]) -> SectionPlan {
    let mut groups = Vec::new();
    let mut run_start = 0;
    for at in 1..=kinds.len() {
        let run_ends = at == kinds.len() || kinds[at] != kinds[run_start];
        if run_ends {
            let size = at - run_start;
            groups.push(MarkedGroup {
                layer: run_start + usize::from(size > 1),
                size,
            });
            run_start = at;
        }
    }
    SectionPlan { groups }
}

impl SectionPlan {
    /// Reconciles the marked readings of one replay into extrapolated sections. Every reading
    /// must be finite and non-negative, the whole token strictly positive, and one reading per
    /// marked group; anything else is refused rather than zipped or defaulted.
    pub(super) fn reconcile(
        &self,
        embed_ms: f32,
        tail_ms: f32,
        group_ms: &[f32],
        whole_ms: f32,
    ) -> Result<SectionTimes, EngError> {
        let check = |what: &str, ms: f32| -> Result<(), EngError> {
            if !ms.is_finite() || ms < 0.0 {
                return Err(EngError::BadRequest(format!(
                    "section timing: the {what} reading is not a finite non-negative time"
                )));
            }
            Ok(())
        };
        check("embed", embed_ms)?;
        check("tail", tail_ms)?;
        if group_ms.len() != self.groups.len() {
            return Err(EngError::BadRequest(format!(
                "section timing: {} group readings for {} marked groups",
                group_ms.len(),
                self.groups.len()
            )));
        }
        for (at, ms) in group_ms.iter().enumerate() {
            check(&format!("group {at}"), *ms)?;
        }
        check("whole token", whole_ms)?;
        if whole_ms <= 0.0 {
            return Err(EngError::BadRequest(
                "section timing: the whole-token reading is not positive".to_owned(),
            ));
        }
        // The derived totals must stay finite too: pathological but finite inputs (huge
        // sections, a tiny whole) can overflow the scaled sum or its ratio, and every reader
        // of `SectionTimes` treats the fields as validated. Built first, then checked, so
        // the methods' own formulas are the single source of the check.
        let times = SectionTimes {
            embed_ms,
            tail_ms,
            group_ms: group_ms.to_vec(),
            sizes: self.groups.iter().map(|group| group.size).collect(),
            whole_ms,
        };
        if !times.extrapolated_ms().is_finite() || !times.whole_ratio().is_finite() {
            return Err(EngError::BadRequest(
                "section timing: the derived totals are not finite".to_owned(),
            ));
        }
        Ok(times)
    }
}

impl SectionTimes {
    /// The embedding section as read, in milliseconds.
    pub(super) fn embed_ms(&self) -> f32 {
        self.embed_ms
    }

    /// The tail section as read (`FinalNorm`..`Advance`), in milliseconds.
    pub(super) fn tail_ms(&self) -> f32 {
        self.tail_ms
    }

    /// The whole token as read (the pair around `Begin`..`Advance`), in milliseconds.
    pub(super) fn whole_ms(&self) -> f32 {
        self.whole_ms
    }

    /// The marked layer of each group as read, in mark order, in milliseconds.
    pub(super) fn group_marked_ms(&self) -> &[f32] {
        &self.group_ms
    }
    /// The summed section GPU time: the embed and tail sections as read, every marked group
    /// scaled by its run size. No unmarked remainder is extrapolated — the span between the
    /// whole-token start and the embed mark and the gaps between sections sit in the
    /// whole-token pair but not in this sum, so it leans below the whole token.
    pub(super) fn extrapolated_ms(&self) -> f32 {
        self.embed_ms
            + self.tail_ms
            + self
                .group_ms
                .iter()
                .zip(&self.sizes)
                .map(|(ms, size)| ms * *size as f32)
                .sum::<f32>()
    }

    /// The summed section time over the measured whole-token time. The qualification
    /// target of the packet reads this within 3% of 1 (`B-kernels-moe-attention-quant.md:71`);
    /// it leans below 1 by the gaps named on `extrapolated_ms`.
    pub(super) fn whole_ratio(&self) -> f32 {
        self.extrapolated_ms() / self.whole_ms
    }

    /// The extrapolated total of each marked run, in mark order.
    pub(super) fn group_totals_ms(&self) -> Vec<f32> {
        self.group_ms
            .iter()
            .zip(&self.sizes)
            .map(|(ms, size)| ms * *size as f32)
            .collect()
    }
}
