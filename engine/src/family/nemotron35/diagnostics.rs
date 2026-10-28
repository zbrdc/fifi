//! Scoped wall diagnostics can overlap; they are not an on-CPU latency partition.

use std::{
    io::Write,
    time::{Duration, Instant},
};

const PREFIX: &str = "nem35_host ";

pub(super) struct HostSpan {
    region: &'static str,
    started: Option<Instant>,
}

impl HostSpan {
    pub(super) fn new(region: &'static str) -> Self {
        Self {
            region,
            started: std::env::var_os("NEM35_PHASE_TIMING").map(|_| Instant::now()),
        }
    }

    pub(super) fn begin_layer(layer: usize, rows: u32) -> Option<Instant> {
        let started = std::env::var_os("NEM35_PHASE_TIMING").map(|_| Instant::now());
        if started.is_some() {
            Self::write_layer_record("begin", layer, rows, None);
        }
        started
    }

    pub(super) fn end_layer(layer: usize, rows: u32, started: Option<Instant>) {
        if let Some(started) = started {
            Self::write_layer_record("end", layer, rows, Some(started.elapsed()));
        }
    }

    fn write_layer_record(phase: &str, layer: usize, rows: u32, elapsed: Option<Duration>) {
        let record = serde_json::json!({
            "phase": phase,
            "layer": layer,
            "rows": rows,
            "elapsed_ns": elapsed.map(|duration| duration.as_nanos().to_string()),
        });
        let _ = writeln!(std::io::stderr().lock(), "nem35_layer {record}");
    }
}

impl Drop for HostSpan {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            // Diagnostic I/O must not replace model errors; collectors reject missing records.
            let record = encode_host_record(self.region, started.elapsed());
            let _ = writeln!(std::io::stderr().lock(), "{record}");
        }
    }
}

fn encode_host_record(region: &str, elapsed: Duration) -> String {
    format!(
        "{PREFIX}{}",
        serde_json::json!({
            "region": region,
            "elapsed_ns": elapsed.as_nanos().to_string(),
        })
    )
}

#[cfg(test)]
mod tests {
    use super::{PREFIX, encode_host_record};
    use std::time::Duration;

    fn record(region: &str, elapsed: Duration) -> serde_json::Value {
        let line = encode_host_record(region, elapsed);
        assert!(line.starts_with(PREFIX));
        serde_json::from_str(line.strip_prefix(PREFIX).unwrap()).unwrap()
    }

    #[test]
    fn host_record_preserves_region_and_nanoseconds() {
        let value = record("embedding_read", Duration::from_nanos(123));
        assert_eq!(value["region"], "embedding_read");
        assert_eq!(value["elapsed_ns"], "123");
    }

    #[test]
    fn host_record_retains_zero_duration() {
        let value = record("matrix_validation", Duration::ZERO);
        assert_eq!(value["elapsed_ns"], "0");
    }

    #[test]
    fn host_record_does_not_truncate_large_duration() {
        let elapsed = Duration::new(u64::MAX, 999_999_999);
        let value = record("projection", elapsed);
        assert_eq!(value["elapsed_ns"], elapsed.as_nanos().to_string());
    }
}
