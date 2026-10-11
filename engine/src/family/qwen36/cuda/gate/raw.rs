//! The raw file of one GPU call (schema `cuda-rd-raw/2`, see `a5_eval.py`): JSON assembly from already computed values.
//! Pure host code (U13). Crates: serde_json 1.0.151 (`json!`, `to_string`), sha2 0.10.9 (`Sha256`, `Digest`).

use super::phases::Timing;
use super::rows::RowSummary;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(super) const SCHEMA: &str = "cuda-rd-raw/2";

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One step. A non-finite float becomes JSON null (serde_json writes null for NaN and infinity), which the evaluator refuses.
pub(super) fn step_json(in_token: u32, s: &RowSummary, forced_logit: Option<f32>) -> Value {
    json!({
        "in_token": in_token,
        "argmax": s.argmax,
        "argmax_logit": s.argmax_logit,
        "top32": s.top.iter().map(|&(id, v)| json!([id, v])).collect::<Vec<_>>(),
        "forced_logit": forced_logit,
        "hash": format!("{:016x}", s.hash),
        "lse": s.lse,
    })
}

/// What the loop knows about one run besides the timing: the flags and clocks that REPLY-72 (Q8, Q9) makes checkable.
pub(super) struct RunMeta {
    /// True only if the graph was captured and replayed in this run.
    pub captured: bool,
    /// True only if the graph was instantiated before the timed region started.
    pub graph_ready_before_timed: bool,
    pub analysis_ns: u64,
    pub syncs: u64,
    /// Host writes to the device per token: the single start-block write (lane C Q-C6). The gate checks 1 per token, 128 per run.
    pub host_writes: u64,
    /// CLOCK_REALTIME instants: the prompt pass synchronized, the timed region started and ended. They let the
    /// evaluator pick the card samples that belong to the timed region.
    pub prompt_synced_unix_ns: u64,
    pub timed_start_unix_ns: u64,
    pub timed_end_unix_ns: u64,
}

pub(super) fn run_json(m: &RunMeta, t: &Timing, steps: Vec<Value>) -> Value {
    json!({
        "captured": m.captured,
        "graph_ready_before_timed": m.graph_ready_before_timed,
        "decode_ns": t.decode_ns,
        "phase_ns": {"replay": t.replay_ns, "readback": t.readback_ns, "host": t.host_ns},
        "step_ns": t.step_ns,
        "analysis_ns": m.analysis_ns,
        "syncs": m.syncs,
        "host_writes": m.host_writes,
        "prompt_synced_unix_ns": m.prompt_synced_unix_ns,
        "timed_start_unix_ns": m.timed_start_unix_ns,
        "timed_end_unix_ns": m.timed_end_unix_ns,
        "steps": steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::cuda::gate::rows::summarize;

    #[test]
    fn sha256_matches_the_standard_vector() {
        assert_eq!(SCHEMA, "cuda-rd-raw/2");
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_step_has_exactly_the_fields_the_evaluator_reads() {
        let s = summarize(&[0.5, 2.0, 1.0]);
        let v = step_json(7, &s, Some(2.0));
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "argmax",
                "argmax_logit",
                "forced_logit",
                "hash",
                "in_token",
                "lse",
                "top32"
            ]
        );
        assert_eq!(v["top32"][0], json!([1, 2.0]));
        assert_eq!(v["hash"].as_str().unwrap().len(), 16);
        assert_eq!(v["argmax_logit"], v["forced_logit"]);
    }

    #[test]
    fn non_finite_values_are_written_as_null() {
        let s = summarize(&[1.0, f32::NAN]);
        let v = step_json(0, &s, None);
        assert!(v["lse"].is_null() && v["forced_logit"].is_null());
    }

    #[test]
    fn a_run_carries_the_phases_the_sync_count_the_flags_and_the_clocks() {
        let t = Timing {
            decode_ns: 100,
            replay_ns: 90,
            readback_ns: 4,
            host_ns: 5,
            step_ns: vec![30, 70],
        };
        let m = RunMeta {
            captured: true,
            graph_ready_before_timed: false,
            analysis_ns: 77,
            syncs: 128,
            host_writes: 7,
            prompt_synced_unix_ns: 1,
            timed_start_unix_ns: 2,
            timed_end_unix_ns: 3,
        };
        let v = run_json(&m, &t, vec![]);
        assert_eq!(
            (
                v["captured"].clone(),
                v["decode_ns"].clone(),
                v["syncs"].clone()
            ),
            (json!(true), json!(100), json!(128))
        );
        assert_eq!(
            v["phase_ns"],
            json!({"replay": 90, "readback": 4, "host": 5})
        );
        assert_eq!(v["analysis_ns"], json!(77));
        assert_eq!(v["host_writes"], json!(7)); // its own field, not the sync count
        assert_eq!(v["step_ns"], json!([30, 70]));
        assert_eq!(v["graph_ready_before_timed"], json!(false));
        assert_eq!(v["captured"], json!(true));
        assert_eq!(
            (
                v["prompt_synced_unix_ns"].clone(),
                v["timed_start_unix_ns"].clone(),
                v["timed_end_unix_ns"].clone()
            ),
            (json!(1), json!(2), json!(3))
        );
    }
}
