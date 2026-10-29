//! The cold-load phase harness of the served Q36 Mac path.
//!
//! The turn-0 cost of a fresh engine (the W34/W35 receipts: 4.7-6.7 s of turn-0 TTFT on a
//! cache miss, identical across cache-limit arms) decomposes into phases this harness walls
//! off and reports as JSON, so the load-lever question (is the wall the disk read, the Metal
//! materialisation, or the first evaluation) is answered by measurement and not by argument:
//!
//! | phase | wall | what it covers |
//! |---|---|---|
//! | `open_s` | checkpoint `open` | the index JSON, every shard header, the consistency checks |
//! | `init_s` | `MlxKernels::try_new` + cache limit | the Metal device, the stream, the lease, the program registry |
//! | `host_load_s` | `HostWeights::load` | every positioned read of the descriptor pass (norms, projections, expert descriptors), timed and byte-counted by the read accounting |
//! | `import_s` | `ResidentDecoder::new_with_kv` | the resident import: the queued batch, the expert stacks' page-aligned reads (also read-accounted), the layer synchronizations |
//! | `first_prefill_s` | one 16-token prompt | the first prompt-mode evaluation of the loaded graph (the prompt-shape first-use JIT; no decode step runs, and the flash-prefill shader is warmed inside the measured import) |
//!
//! The read accounting (`crate::weights::acquire_read::{set, reset, snapshot}`) brackets
//! exactly the open-to-import load phases, so the `read` rows report each phase's bytes,
//! wall time, call count, and effective rate while nothing accumulates during the first
//! evaluation or steady-state serving.
//!
//! This harness is a diagnostic, not a regression gate: empty counters or absurd numbers
//! fail nothing - the JSON numbers are the product, and the reader judges them.
//!
//! Inputs by environment, the contract of `decode_timing`:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `Q36_OUT_DIR` | the report directory, which must not exist | required |
//!
//! The activation pairing and cache form are pinned to the served configuration
//! (`bf16` over `q8`), not read from the environment: the harness's `f32` default is not
//! a served configuration - the first prefill panics on a convert-dtype error - and the
//! phase walls mean the served engine's load.
//!
//! The run (from `engine/`, on the Mac):
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> Q36_OUT_DIR=<OUT_DIR> \
//!   cargo test --release --features native-mlx --lib -- \
//!   family::qwen36::resident::load_timing::cold_load_phase_timing --exact --ignored --nocapture
//! ```
//!
//! The report states the config hash like `decode_timing`; no timing is asserted anywhere.

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use {
    super::decoder::{ResidentDecoder, StepMode},
    super::parity::{CONFIG_SHA256, sha256},
    super::real::{env_path, open, rss_kb},
    super::tool_forward::{CACHE_LIMIT, CONTEXT_LIMIT, SESSION_CAP},
    crate::family::{
        qwen36::{
            host_load::HostWeights,
            settings::{Activation, KvDtype, activation_name},
        },
        spec::QWEN3_6_35B_A3B,
    },
    crate::kernels::MlxKernels,
    crate::sched::SessionBudget,
    crate::weights,
    std::time::Instant,
};

/// The cold-load phase walls and each phase's read accounting, as one JSON object.
///
/// `open` through `first_prefill` are plain walls around the production calls; the `*_read`
/// rows are the accounting snapshots reset at each phase boundary, so a phase that reads
/// nothing reports zeros rather than inheriting the previous phase's bytes.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn read_row() -> serde_json::Value {
    let (bytes, ns, calls) = weights::acquire_read::snapshot();
    weights::acquire_read::reset();
    serde_json::json!({
        "bytes": bytes,
        "wall_s": ns as f64 / 1e9,
        "calls": calls,
        "effective_mib_s": if ns > 0 { bytes as f64 / (ns as f64 / 1e9) / (1024.0 * 1024.0) } else { 0.0 },
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn phase_row(name: &str, wall_s: f64, rss_kib: u64) -> (String, serde_json::Value) {
    (
        name.to_string(),
        serde_json::json!({
            "wall_s": wall_s,
            "read": read_row(),
            "rss_kib_after": rss_kib,
        }),
    )
}

/// Enables the read accounting for its scope: any exit path - completion, panic, or the
/// test-lock poisoning recovery - drops the guard and disables the counters, so a failed
/// phase cannot leave later tests accumulating reads into the same numbers.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct ReadAccountingGuard;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl ReadAccountingGuard {
    fn new() -> Self {
        weights::acquire_read::set(true);
        ReadAccountingGuard
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl Drop for ReadAccountingGuard {
    fn drop(&mut self) {
        weights::acquire_read::set(false);
    }
}

/// The ignored Mac-only cold-load harness: one full engine load from the pinned checkpoint,
/// walled per phase, reported as JSON. No asserts beyond the config pin; the numbers are the
/// product.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "the cold-load phase harness: a real-checkpoint Mac run"]
fn cold_load_phase_timing() {
    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let out_dir = std::path::PathBuf::from(env_path("Q36_OUT_DIR"));
    assert!(
        !out_dir.exists(),
        "Q36_OUT_DIR must not exist: {}",
        out_dir.display()
    );
    std::fs::create_dir_all(&out_dir).unwrap();

    let _accounting = ReadAccountingGuard::new();
    let t = Instant::now();
    let real = open(&ckpt_dir);
    let config_sha = sha256(&real.config);
    assert_eq!(
        config_sha, CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let open_s = t.elapsed().as_secs_f64();
    let open_row = phase_row("open", open_s, rss_kb());

    let t = Instant::now();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let init_s = t.elapsed().as_secs_f64();
    let init_row = phase_row("init", init_s, rss_kb());

    let t = Instant::now();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let host_load_s = t.elapsed().as_secs_f64();
    let host_load_row = phase_row("host_load", host_load_s, rss_kb());

    // The served pairing (bf16 activations over the Q8 cache), pinned: the harness's f32
    // default is not a valid served configuration - `first_prefill` panics on a convert-dtype
    // error under it.
    let activation = Activation::Bf16;
    let kv = KvDtype::Q8;
    let t = Instant::now();
    let mut decoder =
        ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, CONTEXT_LIMIT, activation, kv)
            .unwrap();
    decoder.budget = SessionBudget::new(SESSION_CAP);
    let import_s = t.elapsed().as_secs_f64();
    let import_row = phase_row("import", import_s, rss_kb());
    drop(_accounting);

    // The first prompt-mode evaluation of the loaded graph: the prompt-shape first-use JIT
    // lands here - no decode step runs, and the flash-prefill shader was already warmed
    // inside the measured import.
    let t = Instant::now();
    let ids: Vec<u32> = (0..16u32).cycle().take(16).collect();
    let mut state = decoder.zero_state().unwrap();
    decoder.prefill(&mut state, &ids, StepMode::Prompt).unwrap();
    let first_prefill_s = t.elapsed().as_secs_f64();
    let first_prefill_row = phase_row("first_prefill", first_prefill_s, rss_kb());

    let report = serde_json::json!({
        "config_sha256": config_sha,
        "activation": activation_name(activation),
        "kv_dtype": format!("{kv:?}"),
        "phases": [open_row, init_row, host_load_row, import_row, first_prefill_row],
        "total_s": open_s + init_s + host_load_s + import_s + first_prefill_s,
    });
    let path = out_dir.join("cold_load_phase_timing.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("cold_load_phase_timing={report}");
    println!("wrote {}", path.display());
}
