//! Prefill-chunk routed experts on the device: the dispatch rule only.
//!
//! Large chunks stream each layer's routed expert weights over PCIe once per
//! layer per chunk (llama.cpp's large-batch offload; the Basalt rule): few-row
//! requests stay on the CPU pool, where per-pair CPU work beats a PCIe pass.

/// Few-row chunks stay on the CPU; larger supported pair grids may amortize a PCIe pass.
#[cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(dead_code)
)]
pub(crate) const STREAM_ROWS: usize = 8;

/// The dispatch decision, pure in the env value for testability: the rows
/// threshold, supported pair grid, and explicit `NEM35_CHUNK_EXPERTS=gpu` opt-in.
#[cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(dead_code)
)]
pub(crate) fn wants_device_stream(rows: usize, pairs: usize, env: Option<&str>) -> bool {
    rows > STREAM_ROWS && (rows..=MAX_STREAM_PAIRS).contains(&pairs) && env == Some("gpu")
}

/// The shared row guard accepts at most this many (row, expert) pairs.
#[cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(dead_code)
)]
pub(crate) const MAX_STREAM_PAIRS: usize = 65_536;

#[cfg(test)]
mod tests {
    use super::{STREAM_ROWS, wants_device_stream};

    #[test]
    fn device_stream_needs_the_opt_in_and_more_than_eight_rows() {
        // The default (unset / "cpu") is the CPU pool at every size.
        assert!(!wants_device_stream(STREAM_ROWS + 1, STREAM_ROWS + 1, None));
        assert!(!wants_device_stream(
            STREAM_ROWS + 1,
            STREAM_ROWS + 1,
            Some("cpu")
        ));
        assert!(!wants_device_stream(
            STREAM_ROWS + 1,
            STREAM_ROWS + 1,
            Some("")
        ));
        // "gpu" streams only above the Basalt threshold.
        assert!(!wants_device_stream(STREAM_ROWS, STREAM_ROWS, Some("gpu")));
        assert!(wants_device_stream(
            STREAM_ROWS + 1,
            STREAM_ROWS + 1,
            Some("gpu")
        ));
        // Anything else is not an opt-in; unknown values never stream.
        assert!(!wants_device_stream(4096, 4096, Some("GPU")));
        assert!(!wants_device_stream(4096, 4096, Some("yes")));
    }

    #[test]
    fn device_stream_falls_back_outside_the_pair_grid() {
        assert!(!wants_device_stream(4096, 4096 * 22, Some("gpu")));
        assert!(wants_device_stream(2978, 2978 * 22, Some("gpu")));
        assert!(wants_device_stream(65_536, 65_536, Some("gpu")));
        assert!(!wants_device_stream(65_536, 65_537, Some("gpu")));
        assert!(!wants_device_stream(9, 8, Some("gpu")));
        assert!(!wants_device_stream(9, 0, Some("gpu")));
        assert!(!wants_device_stream(usize::MAX, usize::MAX, Some("gpu")));
    }
}
