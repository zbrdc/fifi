//! Shared graph and state bookkeeping for the family executors.
//!
//! Promoted from `qwen36::cuda`: the state book (KV row admission
//! and health), the per-length graph table of the K-row body, and the step's
//! mode rule. The old `qwen36::cuda` paths re-export from here.

pub(in crate::family) mod book;
#[cfg(test)]
mod book_tests;
pub(in crate::family) mod krow_graph;

/// How a window of the K-row body runs: replay a captured graph, attempt the
/// capture, or run the eager body.
pub(in crate::family) enum Mode {
    Replay,
    Capture,
    Eager,
}

/// The dispatch rule: replay once a graph exists; attempt the capture only on a warmed step
/// that asked for it and has not already refused one; otherwise run eagerly.
pub(in crate::family) fn next_mode(
    graph_ready: bool,
    warmed: bool,
    capture_on: bool,
    refused: bool,
) -> Mode {
    if graph_ready {
        Mode::Replay
    } else if warmed && capture_on && !refused {
        Mode::Capture
    } else {
        Mode::Eager
    }
}
