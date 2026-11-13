//! One CUDA graph per window length of the K-row body (`draft_verify = krow`).
//!
//! The graph of `K` rows records the whole launch list of the family's K-row step for exactly `K`
//! rows: the launch list is fixed by `K` (grids, row-offset ranges, the snapshot strides), and
//! everything that changes from window to window is a device word the kernels read (`pos`,
//! `kv_len`, the window ids, the router's picks). So a graph serves every window of its length and
//! no other length: [`GraphTable`] keeps one slot per `K` and a lookup for `K` can only return the
//! graph captured for `K` rows.
//!
//! The warm rule is the one-row step's (`next_mode` of this module), applied per `K`. The first
//! window of a
//! length runs the body eagerly: it loads the modules of the kernels it launches, which a capture
//! forbids, and warms the views. The next window of that length captures and launches the graph in
//! the same window (a capture records and does not run). Every later window of that length replays.
//! A refused capture is sticky for the whole step and for every length: later windows run the
//! eager K-row body, which is the same kernels in the same order, so the bytes are the same and
//! only the speed is lost.
//!
//! A failed launch. The graph is the whole body, so a failure of `graph.launch` is judged as a failure
//! after the launch: [`ENTERED_BY_GRAPH`] makes the failure rule poison the state and fail the window
//! closed, never take the step back. That rule is pinned by the constant (`krow_graph_tests`) and by a
//! source pin on `replay_krow` (`krow_step_pin_tests`); the failure after the launch of either body is
//! run on the device by the launch fault of the failure check. No fault is injected at the launch call
//! itself, so a failing graph launch is not run on a device.
//!
//! Counters (test builds; the serving path keeps none): per length, the windows that ran the
//! eager body, the captures, the windows that ran the graph (`replays`; the window that captured
//! launches its graph once and counts there), and the node count of the graph. The host counts the
//! replays: a replay re-runs launches that were counted when they were enqueued or captured.
//!
//! Drop order. A graph bakes in the addresses of the K-row arena, the small K-row buffer, the
//! rollback slots and the state. The field `krow_graphs` of the step is declared before `krow`
//! (the small buffer), `state` and `rollback`, so every graph drops before the buffers it covers;
//! `krow_graph_tests` pins that order on the source.

use super::{Mode, next_mode};
use crate::EngError;

/// The longest window the K-row body serves; the table is sized to it. The
/// family's window cap (Qwen3.6's `QWEN36_MAX_WINDOW_ROWS`) aliases this.
pub(in crate::family) const MAX_ROWS: usize = 8;

/// The shortest window the K-row body serves (one row is the single-row step and its own graph).
pub(in crate::family) const MIN_ROWS: usize = 2;

/// Slots of the table: indexed by the window length, so slots 0 and 1 stay empty.
pub(in crate::family) const SLOTS: usize = MAX_ROWS + 1;

/// Ops counted for a window whose graph launch failed. The graph is the whole body, so a failure
/// of the launch is judged as a failure after the launch (the state is poisoned and the window
/// fails closed), never as a refusal of the first op that takes the step back.
pub(in crate::family) const ENTERED_BY_GRAPH: usize = 2;

/// What the table counts, per window length (test builds).
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GraphStats {
    /// Windows that ran the eager body.
    pub(crate) eager: [u64; SLOTS],
    /// Graphs captured.
    pub(crate) captures: [u64; SLOTS],
    /// Windows that ran the graph, the one that captured it included.
    pub(crate) replays: [u64; SLOTS],
    /// Nodes of the captured graph; zero until captured.
    pub(crate) nodes: [usize; SLOTS],
    /// Device bytes the capture and instantiate took (the drop in free memory), when the device
    /// answered; zero until captured.
    pub(crate) vram_bytes: [u64; SLOTS],
}

#[cfg(test)]
impl GraphStats {
    pub(crate) fn eager_total(&self) -> u64 {
        self.eager.iter().sum()
    }

    pub(crate) fn replay_total(&self) -> u64 {
        self.replays.iter().sum()
    }

    pub(crate) fn capture_total(&self) -> u64 {
        self.captures.iter().sum()
    }
}

/// What a capture reports about its graph (kept in test builds only).
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::family) struct Facts {
    #[cfg(test)]
    pub(in crate::family) nodes: usize,
    #[cfg(test)]
    pub(in crate::family) vram_bytes: u64,
}

/// One graph per window length, with the warm flags and the sticky refusal. Generic over the graph
/// type so the host tests run the rules on a fake.
pub(in crate::family) struct GraphTable<G> {
    graphs: [Option<G>; SLOTS],
    warmed: [bool; SLOTS],
    refused: bool,
    /// Test builds: the graphs are switched off, so windows run the eager body on a step that
    /// could capture (the control of the timing and of the D1 matrix).
    #[cfg(test)]
    off: bool,
    #[cfg(test)]
    stats: GraphStats,
}

impl<G> Default for GraphTable<G> {
    fn default() -> Self {
        Self {
            graphs: std::array::from_fn(|_| None),
            warmed: [false; SLOTS],
            refused: false,
            #[cfg(test)]
            off: false,
            #[cfg(test)]
            stats: GraphStats::default(),
        }
    }
}

/// The slot of a window length, or the refusal of a length the K-row body does not serve.
pub(in crate::family) fn slot_of(rows: usize) -> Result<usize, EngError> {
    if (MIN_ROWS..=MAX_ROWS).contains(&rows) {
        Ok(rows)
    } else {
        Err(EngError::BadRequest(format!(
            "draft_verify krow: a K-row graph serves {MIN_ROWS} to {MAX_ROWS} rows, not {rows}"
        )))
    }
}

impl<G> GraphTable<G> {
    /// The graph captured for exactly `rows` rows, if any.
    pub(in crate::family) fn graph(&self, rows: usize) -> Result<Option<&G>, EngError> {
        Ok(self.graphs[slot_of(rows)?].as_ref())
    }

    /// How the next window of `rows` rows runs. `capture_on` is the step's capture setting.
    pub(in crate::family) fn mode(&self, rows: usize, capture_on: bool) -> Result<Mode, EngError> {
        let at = slot_of(rows)?;
        // Switched off (test builds): the eager body, also for a length that has its graph.
        #[cfg(test)]
        if self.off {
            return Ok(Mode::Eager);
        }
        Ok(next_mode(
            self.graphs[at].is_some(),
            self.warmed[at],
            capture_on,
            self.refused,
        ))
    }

    /// Stores the graph just captured for `rows` rows. A second graph for a length is refused: a
    /// length is captured once.
    pub(in crate::family) fn store(
        &mut self,
        rows: usize,
        graph: G,
        _facts: Facts,
    ) -> Result<(), EngError> {
        let at = slot_of(rows)?;
        if self.graphs[at].is_some() {
            return Err(EngError::BadRequest(format!(
                "draft_verify krow: the graph of {rows} rows is already captured"
            )));
        }
        self.graphs[at] = Some(graph);
        #[cfg(test)]
        {
            self.stats.captures[at] += 1;
            self.stats.nodes[at] = _facts.nodes;
            self.stats.vram_bytes[at] = _facts.vram_bytes;
        }
        Ok(())
    }

    /// A window of `rows` rows enqueued the eager body whole: the length is warm.
    pub(in crate::family) fn note_eager(&mut self, rows: usize) {
        if let Ok(at) = slot_of(rows) {
            self.warmed[at] = true;
            #[cfg(test)]
            {
                self.stats.eager[at] += 1;
            }
        }
    }

    /// A window of `rows` rows launched its graph (replay, or the launch after the capture).
    pub(in crate::family) fn note_replay(&mut self, rows: usize) {
        #[cfg(test)]
        if let Ok(at) = slot_of(rows) {
            self.stats.replays[at] += 1;
        }
        #[cfg(not(test))]
        let _ = rows;
    }

    /// The sticky refusal flag, for `fold_capture`.
    pub(in crate::family) fn refused_mut(&mut self) -> &mut bool {
        &mut self.refused
    }

    /// Whether a capture was refused on this step.
    #[cfg(test)]
    pub(crate) fn refused(&self) -> bool {
        self.refused
    }

    /// Switches the graphs off (`false`) or on, for the eager control. Graphs already captured
    /// stay and are used again once the table is switched on.
    #[cfg(test)]
    pub(crate) fn set_on(&mut self, on: bool) {
        self.off = !on;
    }

    /// Whether the graphs are switched on.
    #[cfg(test)]
    pub(crate) fn is_on(&self) -> bool {
        !self.off
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> &GraphStats {
        &self.stats
    }
}

#[cfg(test)]
#[path = "krow_graph_tests.rs"]
mod tests;
