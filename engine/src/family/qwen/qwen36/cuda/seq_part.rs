//! The part of the resident step that belongs to one sequence.
//!
//! One step serves several sessions by time slicing: the weights, the context, the stream, the
//! activation arena, the prefill arena and the prefix store are the step's own and shared. What a
//! sequence owns is its KV planes, its DeltaNet and convolution state and step words (all in
//! [`ResidentState`]), the graph captured over those buffers, and the per-job reports. A
//! [`SeqPart`] holds exactly that. `swap_seq` exchanges it with the part the step runs, a move of
//! a few handles with no device work, so a step that serves one session never swaps and runs the
//! code it always ran.
//!
//! A captured graph bakes in the addresses of the state it ran over, so the graph travels with the
//! state: the field order of [`SeqPart`] drops the graph first.

use super::{
    super::{
        dims::StepOptions,
        run_report::{PrefillReport, RunReport},
        state::ResidentState,
    },
    QwenResidentStep,
};
use crate::{EngError, family::decode_step::state_layout::StateLayout, kernels::cuda::GraphExec};

/// One sequence's state on the device and the bookkeeping that follows it.
pub(crate) struct SeqPart<'a> {
    // Declaration order is drop order: the graph that bakes in the state's addresses goes first.
    graph: Option<GraphExec<'a>>,
    state: ResidentState,
    warmed: bool,
    needs_reset: bool,
    report: RunReport,
    prefill_report: PrefillReport,
}

impl SeqPart<'_> {
    /// The KV cells the sequence's planes hold.
    pub(crate) fn cells(&self) -> usize {
        self.state.layout().kv.capacity
    }
}

impl<'a> QwenResidentStep<'a> {
    /// Whether this step can serve more than one session: the plain resident step only. The
    /// paged, sparse, tiered, banked, K-row, rollback and speculative forms keep state outside the
    /// part a session owns.
    pub(crate) fn check_sessions(&self) -> Result<(), EngError> {
        let refused = if self.res.paged.is_some() {
            Some("paged KV")
        } else if self.res.sparse.is_some() {
            Some("sparse decode")
        } else if self.tiered_mode {
            Some("tiered experts")
        } else if self.exl3_bank.is_some() {
            Some("banked EXL3 experts")
        } else if self.res.krow.is_some() || self.krow_wanted {
            Some("the K-row verify window")
        } else if self.rollback_wanted || self.rollback.is_some() {
            Some("device rollback")
        } else if self.speculator.is_some() {
            Some("speculative decoding")
        } else {
            None
        };
        match refused {
            Some(what) => Err(EngError::BadRequest(format!(
                "engine config: max_seqs above 1 needs the plain resident step, not {what}"
            ))),
            None => Ok(()),
        }
    }

    /// A fresh sequence part whose KV planes hold `cells` positions, zeroed and ready to run. The
    /// stream is the step's own, so the part's first token orders after its own zero fill.
    pub(crate) fn new_seq(&mut self, cells: usize) -> Result<SeqPart<'a>, EngError> {
        self.check_sessions()?;
        let options = self.seq_options(cells);
        let state = ResidentState::allocate(self.context, &self.stream, &self.res.dims, &options)?;
        let mut part = SeqPart {
            graph: None,
            state,
            warmed: false,
            needs_reset: false,
            report: RunReport::default(),
            prefill_report: PrefillReport::default(),
        };
        // The report of a part carries the step's chunk facts, as the build's own does.
        self.swap_seq(&mut part);
        self.refill_prefill_report();
        self.swap_seq(&mut part);
        Ok(part)
    }

    /// Exchanges the part the step runs with `part`. Device work already queued for the part
    /// that leaves stays ordered on the one stream, ahead of anything the part that arrives runs.
    pub(crate) fn swap_seq(&mut self, part: &mut SeqPart<'a>) {
        std::mem::swap(&mut self.graph, &mut part.graph);
        std::mem::swap(&mut self.state, &mut part.state);
        std::mem::swap(&mut self.warmed, &mut part.warmed);
        std::mem::swap(&mut self.needs_reset, &mut part.needs_reset);
        std::mem::swap(&mut self.report, &mut part.report);
        std::mem::swap(&mut self.prefill_report, &mut part.prefill_report);
    }

    /// The options of a sequence part whose KV planes hold `cells` positions.
    fn seq_options(&self, cells: usize) -> StepOptions {
        let layout = self.state.layout();
        StepOptions {
            kv_mode: layout.kv.mode,
            context_tokens: cells,
            delta_slots: layout.delta.slots,
            forked_experts: self.forked_experts,
            tiered: None,
            layout: self.layout,
        }
    }

    /// Device bytes of a sequence part whose KV planes hold `cells` positions.
    pub(crate) fn seq_bytes_at(&self, cells: usize) -> u64 {
        StateLayout::new(&self.res.dims, &self.seq_options(cells))
            .and_then(|layout| layout.bytes())
            .map_or(u64::MAX, |bytes| bytes as u64)
    }

    /// Free device memory now, or `None` when the card cannot be read.
    pub(crate) fn free_device_bytes(&self) -> Option<u64> {
        self.context.memory_info().ok().map(|info| info.free as u64)
    }

    /// The KV cells the running part holds.
    pub(crate) fn seq_cells(&self) -> usize {
        self.state.layout().kv.capacity
    }
}

impl crate::family::pooled_sessions::PartHost for QwenResidentStep<'static> {
    type Part = SeqPart<'static>;
    const LABEL: &'static str = "q36 cuda";

    fn check_sessions(&self) -> Result<(), EngError> {
        QwenResidentStep::check_sessions(self)
    }

    fn context_cells(&self) -> usize {
        self.seq_cells()
    }

    fn new_part(&mut self, cells: usize) -> Result<Self::Part, EngError> {
        self.new_seq(cells)
    }

    fn swap_part(&mut self, part: &mut Self::Part) {
        self.swap_seq(part);
    }

    fn part_cells(part: &Self::Part) -> usize {
        part.cells()
    }

    fn part_bytes_at(&self, cells: usize) -> u64 {
        self.seq_bytes_at(cells)
    }

    fn free_device_bytes(&self) -> Option<u64> {
        QwenResidentStep::free_device_bytes(self)
    }

    fn finish_save(&mut self) -> Result<(), EngError> {
        if self.pending_prefix.is_none() {
            return Ok(());
        }
        self.complete_prefix_save()
    }

    fn discard_save(&mut self) {
        self.drop_pending_prefix();
    }
}
