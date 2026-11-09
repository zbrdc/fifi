//! The sequence, the position, the poison rule and the chunk accounting of the chunk driver, on
//! fakes.

use super::ChunkDriver;
use crate::{
    EngError,
    family::{flash_next::cuda::driver::fakes, session::TokenStepper},
};
use fakes::{FakeChunkDevice, FakeFeeds, LOGITS, SAMPLER_VOCAB, STREAM, shape};
use crate::family::flash_next::cuda::ple_feed::PleKeyValue;

fn chunk_driver() -> ChunkDriver<FakeChunkDevice, FakeFeeds> {
    ChunkDriver::new(FakeChunkDevice::default(), FakeFeeds::default(), shape()).unwrap()
}

#[test]
fn a_fed_chunk_runs_the_start_the_layers_around_one_ple_roundtrip_and_no_head() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(
        driver.device().log,
        [
            "chunk_params 3 0 2",
            "chunk_stream",
            "chunk_ple",
            "begin",
            "layers 0..2 @0 m2",
            "read_stream",
            "write_back",
            "layers 2..4 @0 m2",
            "health",
        ]
    );
    assert_eq!(driver.fed(), 2);
    // Each row's update saw that row's stream after layers 0 and 1: every layer adds
    // layer + 1 + pos + row, so row 0 (token 3) is 3+1+2=6 and row 1 (token 4) is 4+2+3=9.
    assert_eq!(
        driver.feeds().updates,
        [(3, vec![6.0; STREAM]), (4, vec![9.0; STREAM]),]
    );
}

#[test]
fn the_logits_chunk_runs_the_head_over_the_last_row() {
    let mut driver = chunk_driver();
    let row = driver.run_chunk(&[3, 4], true).unwrap().unwrap();
    let log = &driver.device().log;
    assert_eq!(
        &log[log.len() - 3..],
        ["head", "read_logits", "health"].map(String::from)
    );
    assert_eq!(row.len(), LOGITS);
    // The last row (token 4, row 1) after every layer, plus its 0.5 update: the layers add
    // 2+3 before the PLE roundtrip and 4+5 after, so 4 + 5 + 9 + 0.5 = 18.5 at every value.
    for (i, value) in row.iter().enumerate() {
        assert_eq!(*value, 18.5 * i as f32);
    }
}

#[test]
fn the_device_ple_route_projects_on_the_device_and_hands_the_pairs_over() {
    let mut driver = chunk_driver();
    driver.ple_device = true;
    driver.device_mut().ple_pairs = Some(
        (0..2)
            .map(|r| PleKeyValue {
                keys: vec![r as f32; STREAM],
                value: vec![r as f32; 1],
            })
            .collect(),
    );
    driver.run_chunk(&[3, 4], false).unwrap();
    assert!(
        driver
            .device()
            .log
            .iter()
            .any(|call| call.starts_with("chunk_ple_project"))
    );
    // The pairs reach the step in row order.
    assert_eq!(driver.feeds().pairs, [0.0, 1.0]);
    // The plain route never projects on the device.
    let mut plain = chunk_driver();
    plain.run_chunk(&[3, 4], false).unwrap();
    assert!(!plain
        .device()
        .log
        .iter()
        .any(|call| call.starts_with("chunk_ple_project")));
    assert!(plain.feeds().pairs.is_empty());
}

#[test]
fn the_device_route_without_a_projection_is_refused_and_poisons() {
    let mut driver = chunk_driver();
    driver.ple_device = true;
    // The fake has no pairs, so `ple_project_chunk` fails; the driver must poison, not limp on.
    assert!(driver.run_chunk(&[3], false).is_err());
    assert!(driver.feed(4).is_err());
}

#[test]
fn the_position_advances_by_the_chunk_and_reaches_the_layers() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[1, 2, 3], false).unwrap();
    driver.run_chunk(&[4], true).unwrap();
    let log = &driver.device().log;
    assert!(log.contains(&"layers 0..2 @3 m1".to_string()));
    assert_eq!(driver.fed(), 4);
}

#[test]
fn a_chunk_that_would_overrun_the_caches_is_refused_without_writes() {
    let mut driver = chunk_driver();
    driver.run_chunk(&[1, 2, 3, 4], false).unwrap();
    // 4 + 3 > POSITIONS (6): refused before any device write, nothing poisoned.
    let error = driver.run_chunk(&[5, 6, 7], false).unwrap_err();
    assert!(matches!(error, EngError::ContextLimit(_)));
    assert_eq!(driver.fed(), 4);
    assert_eq!(driver.device().log.last().unwrap(), "health");
    driver.run_chunk(&[5, 4], false).unwrap();
    assert_eq!(driver.fed(), 6);
}

#[test]
fn an_empty_chunk_and_an_out_of_range_token_change_nothing() {
    let mut driver = chunk_driver();
    assert!(driver.run_chunk(&[], false).is_err());
    assert!(driver.run_chunk(&[3, SAMPLER_VOCAB as u32], false).is_err());
    assert!(driver.device().log.is_empty());
    assert_eq!(driver.fed(), 0);
}

#[test]
fn a_failure_after_the_first_write_poisons_until_reset() {
    let device = FakeChunkDevice {
        fail: Some("layers"),
        ..FakeChunkDevice::default()
    };
    let mut driver = ChunkDriver::new(device, FakeFeeds::default(), shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    // The chunk did not land: the position stayed, and the poison refuses later chunks.
    assert_eq!(driver.fed(), 0);
    assert!(driver.run_chunk(&[3], false).is_err());
    driver.reset().unwrap();
    driver.run_chunk(&[3, 4], false).unwrap();
    assert_eq!(driver.fed(), 2);
}

#[test]
fn a_failed_host_begin_leaves_the_state_unpoisoned() {
    let feeds = FakeFeeds {
        fail_begin: Some(4),
        ..FakeFeeds::default()
    };
    let mut driver = ChunkDriver::new(FakeChunkDevice::default(), feeds, shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    assert_eq!(driver.fed(), 0);
    // A host-read failure is not a device failure: the next chunk still runs.
    driver.run_chunk(&[3], false).unwrap();
    assert_eq!(driver.fed(), 1);
}

#[test]
fn a_ple_update_of_the_wrong_width_is_refused_and_poisons() {
    let feeds = FakeFeeds {
        update_width: Some(STREAM + 1),
        ..FakeFeeds::default()
    };
    let mut driver = ChunkDriver::new(FakeChunkDevice::default(), feeds, shape()).unwrap();
    assert!(driver.run_chunk(&[3, 4], false).is_err());
    assert_eq!(driver.fed(), 0);
    assert!(driver.run_chunk(&[3], false).is_err());
}

#[test]
fn the_shape_refuses_a_ple_layer_at_or_past_the_layers() {
    let mut bad = shape();
    bad.ple_layer = bad.layers;
    assert!(ChunkDriver::new(FakeChunkDevice::default(), FakeFeeds::default(), bad).is_err());
}

#[test]
fn feed_rows_splits_the_prompt_into_chunks_of_the_cap() {
    let mut driver = chunk_driver();
    driver.chunk_cap = 3;
    driver.feed_rows(&[1, 2, 3, 4, 5]).unwrap();
    let params: Vec<_> = driver
        .device()
        .log
        .iter()
        .filter(|line| line.starts_with("chunk_params"))
        .cloned()
        .collect();
    assert_eq!(
        params,
        ["chunk_params 1 0 3", "chunk_params 4 3 2"].map(String::from)
    );
    assert_eq!(driver.fed(), 5);
}

#[test]
fn feed_rows_with_the_default_cap_feeds_one_token_at_a_time() {
    let mut driver = chunk_driver();
    driver.feed_rows(&[1, 2]).unwrap();
    let params: Vec<_> = driver
        .device()
        .log
        .iter()
        .filter(|line| line.starts_with("chunk_params"))
        .cloned()
        .collect();
    assert_eq!(
        params,
        ["chunk_params 1 0 1", "chunk_params 2 1 1"].map(String::from)
    );
}
