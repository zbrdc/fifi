//! The Qwen3.8 Flash-Next CUDA path (`docs/plans/flash-next-cuda.md`).
//!
//! - Loading: `load` (the trunk tensors from the pack walk) and the host feeds `embedding`,
//!   `ple_feed`, `feeds`, `pack_feeds` and `stream_io`.
//! - State: `state` and `step_plan` (the device state and the decode arena plan).
//! - Expert placement: `bank` (device slots, pinned host store and the slot policy), `bank_source`
//!   (the records from the pack) and `moe_block` (the MoE block split over the bank).
//! - The decode step: `step`, `mixer`, `driver`, `executor`, `device` and `start`.
//!
//! Each module's own header says what it does.

mod bank;
// The pack reader exists where the pack files do (`load::pack_files`): tests and native-cuda Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod bank_source;
#[cfg(test)]
mod bank_source_tests;
#[cfg(test)]
mod bank_tests;
mod moe_block;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod moe_block_device_tests;
#[cfg(test)]
mod moe_block_tests;
mod prefill_ring;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod prefill_ring_device_tests;
#[cfg(test)]
mod prefill_ring_tests;
// S3: the state and decode arena plans size device memory with the kernel wrappers, so they build
// with the native CUDA backend only. Staged: the decode step (S4) is the first caller.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for the decode step (S4)")
)]
mod state;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod state_device_tests;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod state_tests;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for the decode step (S4)")
)]
mod step_plan;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_plan_hazard_tests;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_plan_tests;
// The trunk list needs the pack walk (`load::exl3_pack`), which exists in tests and on native-cuda
// Linux.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod embedding;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod feeds;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(in crate::family::flash_next) mod load;
#[cfg(test)]
mod load_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod ple_feed;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod stream_io;

#[cfg(test)]
mod testkit {
    //! An in-memory safetensors checkpoint for the host tests of the feeds.

    use crate::weights::{ShardIo, StreamedCheckpoint};

    /// One shard holding `(name, dtype, shape, bytes)` tensors.
    pub(super) fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }
}

// The decode step. It needs the state and step plans, so it builds with the native CUDA backend
// only. Staged: the executor is the first caller.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod mixer;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod step;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_tests;

// The token driver and the executor on top of it. The driver runs against fakes in host tests;
// the card implementations and the engine start need the native CUDA backend.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
mod device;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(not(test), allow(dead_code, reason = "staged for the engine start"))]
pub(in crate::family::flash_next) mod driver;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod driver_device_tests;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(not(test), allow(dead_code, reason = "staged for the engine start"))]
mod executor;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(not(test), allow(dead_code, reason = "staged for the engine start"))]
mod pack_feeds;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::flash_next) mod start;
