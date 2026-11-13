//! Host tests of the per-length graph table: which graph a window of `K` rows can use, the warm
//! rule per length, and the counters. The table is generic over the graph type, so a plain
//! integer stands for a graph here and the real `GraphExec` is exercised on the device
//! (`d1_window_device_tests`). The tests bound to Qwen3.6 (the sticky-refusal fold, the failure
//! rule, the step-struct drop order) live beside `qwen36::cuda::step_tests`.

use super::{Facts, GraphTable, MAX_ROWS, MIN_ROWS, Mode, SLOTS, slot_of};

fn facts(nodes: usize, vram_bytes: u64) -> Facts {
    Facts { nodes, vram_bytes }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Replay => "replay",
        Mode::Capture => "capture",
        Mode::Eager => "eager",
    }
}

fn mode(table: &GraphTable<usize>, rows: usize, capture_on: bool) -> &'static str {
    mode_name(table.mode(rows, capture_on).expect("a served length"))
}

/// The rollback window cap is the table capacity: two names, one number, pinned here.
#[test]
fn the_rollback_cap_is_the_table_capacity() {
    use crate::family::qwen36::rollback;
    assert_eq!(rollback::MAX_ROWS, MAX_ROWS);
}

#[test]
fn only_windows_of_two_to_the_cap_have_a_slot() {
    assert_eq!(SLOTS, MAX_ROWS + 1);
    for rows in [0, 1, MAX_ROWS + 1, 64, usize::MAX] {
        assert!(slot_of(rows).is_err(), "{rows} rows");
    }
    for rows in MIN_ROWS..=MAX_ROWS {
        assert_eq!(slot_of(rows).expect("served"), rows);
    }
    let table = GraphTable::<usize>::default();
    assert!(table.graph(1).is_err());
    assert!(table.mode(9, true).is_err());
}

/// The mutant to kill: the graph of `K` rows reused for `K + 1` rows. With one graph per length, storing the
/// graph of a length never makes another length ready, and a lookup returns the graph stored for
/// exactly that length.
#[test]
fn the_graph_of_k_rows_is_never_the_graph_of_another_length() {
    let mut table = GraphTable::<usize>::default();
    for rows in MIN_ROWS..=MAX_ROWS {
        table.note_eager(rows);
        // A graph labelled by its own length.
        table
            .store(rows, rows, facts(100 + rows, 0))
            .expect("store");
        for other in MIN_ROWS..=MAX_ROWS {
            let expected = (other <= rows).then_some(other);
            assert_eq!(
                table.graph(other).expect("served").copied(),
                expected,
                "after storing {rows}: the graph of {other} rows"
            );
        }
    }
    for rows in MIN_ROWS..=MAX_ROWS {
        assert_eq!(table.graph(rows).expect("served").copied(), Some(rows));
        assert_eq!(mode(&table, rows, true), "replay");
    }
}

#[test]
fn a_length_without_a_graph_is_not_served_by_the_graph_of_the_length_below() {
    let mut table = GraphTable::<usize>::default();
    table.note_eager(3);
    table.store(3, 3, facts(0, 0)).expect("store");
    assert_eq!(mode(&table, 3, true), "replay");
    // 4 has no graph, is not warm, and must run eagerly first: not a replay of the 3-row graph.
    assert_eq!(mode(&table, 4, true), "eager");
    assert!(table.graph(4).expect("served").is_none());
    assert_eq!(mode(&table, 2, true), "eager");
}

#[test]
fn each_length_runs_eager_once_then_captures_then_replays() {
    let mut table = GraphTable::<usize>::default();
    for rows in MIN_ROWS..=MAX_ROWS {
        assert_eq!(mode(&table, rows, true), "eager", "{rows}: first window");
        table.note_eager(rows);
        assert_eq!(mode(&table, rows, true), "capture", "{rows}: second window");
        // The other lengths are not affected by this one.
        for other in (MIN_ROWS..=MAX_ROWS).filter(|&other| other > rows) {
            assert_eq!(
                mode(&table, other, true),
                "eager",
                "{other} while {rows} warm"
            );
        }
        table.store(rows, rows, facts(0, 0)).expect("store");
        assert_eq!(mode(&table, rows, true), "replay", "{rows}: third window");
    }
}

#[test]
fn a_step_that_does_not_capture_never_captures_and_never_replays() {
    let mut table = GraphTable::<usize>::default();
    for rows in MIN_ROWS..=MAX_ROWS {
        assert_eq!(mode(&table, rows, false), "eager");
        table.note_eager(rows);
        assert_eq!(
            mode(&table, rows, false),
            "eager",
            "{rows}: warm but capture off"
        );
    }
}

#[test]
fn a_length_is_captured_once() {
    let mut table = GraphTable::<usize>::default();
    table.note_eager(5);
    table.store(5, 1, facts(0, 0)).expect("store");
    let error = table
        .store(5, 2, facts(0, 0))
        .expect_err("a second graph for 5 rows");
    assert!(
        format!("{error:?}").contains("already captured"),
        "{error:?}"
    );
    assert_eq!(table.graph(5).expect("served").copied(), Some(1));
}

#[test]
fn the_counters_are_per_length_and_count_the_capturing_window_as_a_launch() {
    let mut table = GraphTable::<usize>::default();
    table.note_eager(4);
    table.store(4, 4, facts(321, 4096)).expect("capture");
    table.note_replay(4);
    table.note_replay(4);
    table.note_eager(2);
    let stats = table.stats();
    assert_eq!(stats.eager[4], 1);
    assert_eq!(stats.eager[2], 1);
    assert_eq!(stats.captures[4], 1);
    assert_eq!(stats.replays[4], 2);
    assert_eq!(stats.nodes[4], 321);
    assert_eq!(stats.vram_bytes[4], 4096);
    for rows in [0, 1, 3, 5, 6, 7, 8] {
        assert_eq!(stats.replays[rows], 0, "{rows}");
        assert_eq!(stats.captures[rows], 0, "{rows}");
    }
    assert_eq!(stats.replay_total(), 2);
    assert_eq!(stats.eager_total(), 2);
    assert_eq!(stats.capture_total(), 1);
}

#[test]
fn switching_the_graphs_off_runs_the_eager_body_and_back_on_replays_the_stored_graph() {
    let mut table = GraphTable::<usize>::default();
    table.note_eager(6);
    table.store(6, 6, facts(0, 0)).expect("store");
    assert_eq!(mode(&table, 6, true), "replay");
    table.set_on(false);
    assert!(!table.is_on());
    assert_eq!(mode(&table, 6, true), "eager");
    assert_eq!(mode(&table, 6, false), "eager");
    table.set_on(true);
    assert_eq!(mode(&table, 6, true), "replay");
}
