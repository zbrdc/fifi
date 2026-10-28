//! What an ANE sidecar classifier costs while the engine decodes (packet ane-sidecar-1).
//!
//! The probe measures the private ANE route of the decision head (`DecisionHead::ane` through
//! `ane_bridge`) and answers one question: what does one head call cost, and what does it do to
//! a real Q36 decode loop? It prints one machine-parseable line per mode, both with the
//! packet's exact key set and JSON nulls where a mode does not measure a field.
//!
//! The threading answer comes first because it shapes the probe. `DecisionHead` is neither
//! `Send` nor `Sync`: the `backend` field is `Box<dyn HeadBackend>` behind the private
//! `HeadBackend` trait, which carries no `Send` bound (engine/src/kernels/decision/mod.rs), so
//! the auto traits do not propagate and no second thread can hold the head. The C++ bridge
//! serializes every entry point through `g_bridge_mutex`, which protects the FFI layer, but
//! that changes nothing for the Rust type. A threaded sidecar therefore needs a design change
//! (a `Send` wrapper or a channel), not a test trick, and the decode mode below is SERIAL: one
//! head call between decode steps, on the same thread, labeled as such. It measures the serial
//! insertion cost, not concurrency.
//!
//! The decode mode reuses the load and prefill path of `decode_timing` (the same
//! `Q36_CKPT_DIR`, the same required `Q36_ACTIVATION`, the optional `Q36_KV_DTYPE`, and the
//! same ids sources: the tool prompt of `IDS_FILE` + `REF_JSON` by default, or the
//! `Q36_PROMPT_IDS` file with `DT_IDS_SOURCE=file`), prefills the whole ids source through the
//! serving path before each timed window, and decodes `DT_TOKENS` greedy tokens per window
//! (default 64). No `Q36_OUT_DIR` is required: the probe prints and writes no artifact. The
//! exact Mac command:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_ACTIVATION=bf16 \
//! cargo test --release --features native-mlx,native-ane --lib \
//!   family::qwen36::resident::ane_latency::ane_head_latency_under_decode \
//!   -- --ignored --exact --nocapture
//! ```

#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
use {
    super::super::settings::{activation_name, chunked_prefill_allowed, kv_dtype_name},
    super::chunk_plan::{ChunkCost, PrefillTerms, admitted_chunk},
    super::decode_timing::{read_prompt_ids, required_activation_from_env},
    super::decoder::{ResidentDecoder, StepMode},
    super::parity::{CONFIG_SHA256, sha256},
    super::real::{env_path, kv_dtype_from_env, open},
    super::state::ResidentState,
    super::tool_forward::{
        CACHE_LIMIT, CONTEXT_LIMIT, NO_BUDGET_PIECE, PREFILL_BUDGET, PREFILL_RESERVE, SESSION_CAP,
        ServeStep, read_ids, serve_chunk_setting,
    },
    super::within_mode::env_number,
    crate::family::{
        qwen36::host_load::HostWeights, session::DecodeSession, spec::QWEN3_6_35B_A3B,
    },
    crate::kernels::MlxKernels,
    crate::kernels::decision::{DecisionHead, HeadConfig, HeadWeights},
    crate::sched::SessionBudget,
    std::time::Instant,
};

/// The timed calls of each idle-mode adapter: the 500 the packet asks for as a minimum.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
const CALLS: usize = 500;
/// The untimed warm-up calls of each idle-mode adapter, dropped from every percentile.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
const WARMUP_CALLS: usize = 16;
/// The untimed warm-up decode tokens before each timed window, as `decode_timing` runs them.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
const WARMUP_TOKENS: usize = 2;
/// The largest `DT_TOKENS` the run accepts, `decode_timing`'s own window bound.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
const MAX_TIMED_TOKENS: usize = 4096;

/// The decision-head shape of the existing ANE gate (`tests/decision_ane.rs`): the bridge
/// accepts it and the parity numbers are measured against it.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn config() -> HeadConfig {
    HeadConfig {
        input: 4096,
        hidden: 512,
        experts: 256,
        classes: 4,
    }
}

/// The synthetic weights of the same gate: the same recipes, so the latency runs on the exact
/// shapes and values the parity gate already qualified.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn weights() -> HeadWeights {
    let c = config();
    let mut w1 = Vec::with_capacity(c.hidden * c.input);
    for i in 0..c.hidden * c.input {
        w1.push((i % 7) as f32 / 10.0 - 0.3);
    }
    let mut w2 = Vec::with_capacity((c.experts + c.classes) * c.hidden);
    for i in 0..(c.experts + c.classes) * c.hidden {
        w2.push((i % 11) as f32 / 100.0 - 0.05);
    }
    let mut b1 = Vec::with_capacity(c.hidden);
    for i in 0..c.hidden {
        b1.push((i % 5) as f32 / 40.0 - 0.025);
    }
    let mut b2 = Vec::with_capacity(c.classes + c.experts);
    for i in 0..c.classes + c.experts {
        b2.push((i % 7) as f32 / 30.0 - 0.1);
    }
    HeadWeights::try_new(&c, w1, b1, w2, b2).unwrap()
}

/// The feature row of the same gate.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn features() -> Vec<f32> {
    (0..config().input)
        .map(|i| (i % 11) as f32 / 10.0)
        .collect()
}

/// The ids of the run's source: the tool prompt of `IDS_FILE` + `REF_JSON` (the default, as
/// `decode_timing`) or the `Q36_PROMPT_IDS` file with `DT_IDS_SOURCE=file`. Any other word is
/// an error naming the variable.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn ids_from_env() -> Vec<u32> {
    match std::env::var("DT_IDS_SOURCE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("tool") => tool_ids(),
        Ok("file") => read_prompt_ids(),
        Ok(other) => panic!("DT_IDS_SOURCE must be tool or file, got {other:?}"),
        Err(error) => panic!("DT_IDS_SOURCE: {error}"),
    }
}

/// The 922-id tool prompt through `tool_forward::read_ids`, the same reader as
/// `decode_timing`'s default source.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn tool_ids() -> Vec<u32> {
    let ids_path = env_path("IDS_FILE");
    let ref_path = env_path("REF_JSON");
    let ids_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
    let ref_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    read_ids(&ids_value, &ref_value)
}

/// The nearest-rank percentile `p` (0 to 100) of `values`, the method `decode_timing`'s own
/// private helper uses (it stays module-private there, so this probe carries its own copy):
/// the smallest value whose rank, `ceil(p/100 * n)`, reaches it. Panics on an empty slice;
/// every caller feeds a non-empty timed window.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn percentile(values: &[f64], p: u32) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p.min(100) as usize * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// A duration in milliseconds, as `decode_timing` reports them.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

/// The probe's only assertion: the figure is finite and positive. No speed is gated anywhere,
/// not even indirectly through a threshold.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn assert_positive(value: f64, what: &str) {
    assert!(
        value.is_finite() && value > 0.0,
        "{what} must be finite and positive, got {value}"
    );
}

/// One adapter's idle latencies: `warmup` untimed `predict` calls, then `calls` timed ones.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn idle_latency(head: &DecisionHead, features: &[f32], warmup: usize, calls: usize) -> Vec<f64> {
    for _ in 0..warmup {
        head.predict(features).unwrap();
    }
    let mut samples = Vec::with_capacity(calls);
    for _ in 0..calls {
        let started = Instant::now();
        head.predict(features).unwrap();
        samples.push(ms(started.elapsed()));
    }
    samples
}

/// The figures of one printed probe line. `None` prints as a JSON null, so each mode prints
/// the packet's exact key set and names what it did not measure.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
struct LineFigures {
    mode: &'static str,
    ane_p50_ms: Option<f64>,
    ane_p99_ms: Option<f64>,
    cpu_p50_ms: Option<f64>,
    decode_tps_with: Option<f64>,
    decode_tps_without: Option<f64>,
}

#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
impl LineFigures {
    /// The line itself, with the packet's keys in its order; `threaded` is always `false`
    /// because the head cannot cross threads (see the module doc).
    fn text(&self) -> String {
        let figure = |value: Option<f64>| {
            value.map_or_else(|| "null".to_string(), |value| format!("{value:.3}"))
        };
        format!(
            "ane_head_latency={{\"mode\":\"{}\",\"ane_p50_ms\":{},\"ane_p99_ms\":{},\
             \"cpu_p50_ms\":{},\"decode_tps_with\":{},\"decode_tps_without\":{},\
             \"threaded\":false}}",
            self.mode,
            figure(self.ane_p50_ms),
            figure(self.ane_p99_ms),
            figure(self.cpu_p50_ms),
            figure(self.decode_tps_with),
            figure(self.decode_tps_without),
        )
    }
}

/// One timed decode window from a fresh state: the serving prefill of `prompt` (the planned
/// chunk and worker piece, exactly as `decode_timing::run_case` feeds it), `WARMUP_TOKENS`
/// untimed greedy tokens, then `timed` greedy tokens. When `probe` is `Some`, one ANE
/// `predict` runs between each pair of timed steps on this same thread (serial, not
/// concurrent) and its per-call walls are returned. The result is the timed window's wall in
/// milliseconds and those call walls.
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn decode_window(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    prompt: &[u32],
    chunk: Option<u32>,
    timed: usize,
    probe: Option<(&DecisionHead, &[f32])>,
) -> (f64, Vec<f64>) {
    decoder.reset(state).unwrap();
    if let Some(rows) = chunk {
        decoder.set_chunk_rows(rows).unwrap();
    }
    let piece = chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    let limit = usize::try_from(CONTEXT_LIMIT).unwrap();
    let mut session = DecodeSession::default();
    let mut fed = Vec::new();
    {
        let mut step = ServeStep {
            decoder,
            state,
            chunk,
            fed: &mut fed,
        };
        for rows in prompt.chunks(piece) {
            session.prefill(&mut step, rows, limit).unwrap();
        }
    }
    let mut input = session.pending().unwrap();
    for _ in 0..WARMUP_TOKENS {
        input = decoder
            .step(state, input, StepMode::Generate)
            .unwrap()
            .token
            .unwrap();
    }
    let mut calls = Vec::new();
    let started = Instant::now();
    for _ in 0..timed {
        input = decoder
            .step(state, input, StepMode::Generate)
            .unwrap()
            .token
            .unwrap();
        if let Some((head, features)) = probe {
            let call_started = Instant::now();
            head.predict(features).unwrap();
            calls.push(ms(call_started.elapsed()));
        }
    }
    (ms(started.elapsed()), calls)
}

/// The ANE sidecar latency probe (packet ane-sidecar-1). Idle mode first: the exact head
/// config and synthetic weights of the existing ANE gate, ANE p50/p90/p99 over `CALLS` timed
/// calls, and the CPU adapter on the same weights and calls. Decode mode second: the same
/// load, prefill, and greedy decode loop as `decode_timing`, timed once without the head and
/// once with one serial head call between each pair of steps. Reports, gates no speed.
#[test]
#[ignore = "needs the real checkpoint and the ANE bridge on the Mac; run with --ignored"]
#[cfg(all(
    feature = "native-mlx",
    feature = "native-ane",
    target_os = "macos",
    target_arch = "aarch64",
))]
fn ane_head_latency_under_decode() {
    let _serial = crate::native_test_lock();

    // Every input is validated first, as `decode_timing` validates, before any measurement.
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let ids = ids_from_env();
    let timed = env_number("DT_TOKENS", 64, 1, MAX_TIMED_TOKENS);
    let activation = required_activation_from_env();

    // Idle mode: no model is loaded yet, only the two adapters on the same weights.
    let c = config();
    let features = features();
    let ane = DecisionHead::ane(weights()).unwrap();
    let cpu = DecisionHead::cpu(weights()).unwrap();
    println!(
        "ane-latency head: input {} hidden {} experts {} classes {}, the existing gate's \
         shape and synthetic weights; {CALLS} timed calls per adapter after {WARMUP_CALLS} \
         warm-up calls",
        c.input, c.hidden, c.experts, c.classes,
    );
    println!(
        "sidecar threading: DecisionHead is neither Send nor Sync (its backend is \
         Box<dyn HeadBackend> behind a private trait with no Send bound, \
         engine/src/kernels/decision/mod.rs), so no second thread can hold the head; the \
         decode mode below is SERIAL: one head call between decode steps on the same thread, \
         not concurrent"
    );
    let ane_idle = idle_latency(&ane, &features, WARMUP_CALLS, CALLS);
    let cpu_idle = idle_latency(&cpu, &features, WARMUP_CALLS, CALLS);
    assert_eq!(ane_idle.len(), CALLS, "every idle ane call must be timed");
    assert_eq!(cpu_idle.len(), CALLS, "every idle cpu call must be timed");
    let ane_p50 = percentile(&ane_idle, 50);
    let ane_p90 = percentile(&ane_idle, 90);
    let ane_p99 = percentile(&ane_idle, 99);
    let cpu_p50 = percentile(&cpu_idle, 50);
    let cpu_p90 = percentile(&cpu_idle, 90);
    let cpu_p99 = percentile(&cpu_idle, 99);
    for (what, value) in [
        ("idle ane p50", ane_p50),
        ("idle ane p90", ane_p90),
        ("idle ane p99", ane_p99),
        ("idle cpu p50", cpu_p50),
        ("idle cpu p90", cpu_p90),
        ("idle cpu p99", cpu_p99),
    ] {
        assert_positive(value, what);
    }
    println!(
        "idle latencies ms: ane p50/p90/p99 {ane_p50:.3}/{ane_p90:.3}/{ane_p99:.3}, cpu \
         p50/p90/p99 {cpu_p50:.3}/{cpu_p90:.3}/{cpu_p99:.3} (the same weights and the same \
         {CALLS} calls)"
    );
    println!(
        "{}",
        LineFigures {
            mode: "idle",
            ane_p50_ms: Some(ane_p50),
            ane_p99_ms: Some(ane_p99),
            cpu_p50_ms: Some(cpu_p50),
            decode_tps_with: None,
            decode_tps_without: None,
        }
        .text()
    );

    // Decode mode: the same load and prefill as `decode_timing`, then two windows over the
    // same prompt from fresh states, the without-head window first, as the lighter case runs
    // first in `decode_timing`'s table.
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let heads_q = u64::try_from(host.shape().attn.heads_q).unwrap();
    let kv = kv_dtype_from_env();
    let mut decoder =
        ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, CONTEXT_LIMIT, activation, kv)
            .unwrap();
    decoder.budget = SessionBudget::new(SESSION_CAP);
    let mut state = decoder.zero_state().unwrap();
    let terms = PrefillTerms {
        budget: PREFILL_BUDGET,
        weights: decoder.weights.byte_len().unwrap(),
        session_reserve: PREFILL_RESERVE,
    };
    let cost = ChunkCost::new(heads_q, CONTEXT_LIMIT).unwrap();
    let serve_setting = serve_chunk_setting(activation);
    let chunked = chunked_prefill_allowed(decoder.weights.consts.act);
    let chunk = if chunked {
        admitted_chunk(&cost, terms.remaining(), serve_setting).unwrap()
    } else {
        None
    };
    println!(
        "ane-latency decode plan: activation {}, kv {}, chunked prefill {chunked}, serving \
         chunk {chunk:?}, {timed} timed greedy tokens per window after {WARMUP_TOKENS} \
         warm-up tokens, prompt {} tokens",
        activation_name(activation),
        kv_dtype_name(kv),
        ids.len(),
    );
    let (without_ms, _) = decode_window(&mut decoder, &mut state, &ids, chunk, timed, None);
    let (with_ms, calls) = decode_window(
        &mut decoder,
        &mut state,
        &ids,
        chunk,
        timed,
        Some((&ane, &features)),
    );
    assert_eq!(
        calls.len(),
        timed,
        "the with-head window must run one head call per timed step"
    );
    let steps = timed as f64;
    let tps_without = 1000.0 * steps / without_ms;
    let tps_with = 1000.0 * steps / with_ms;
    // A difference of window walls, so decode noise rides on it: reported, never asserted
    // positive, because asserting its sign would gate a speed the probe only measures.
    let added_ms_per_step = (with_ms - without_ms) / steps;
    let in_loop_mean = calls.iter().sum::<f64>() / steps;
    let in_loop_p50 = percentile(&calls, 50);
    let in_loop_p99 = percentile(&calls, 99);
    for (what, value) in [
        ("decode tps without the head", tps_without),
        ("decode tps with the head", tps_with),
        ("in-loop ane mean", in_loop_mean),
        ("in-loop ane p50", in_loop_p50),
        ("in-loop ane p99", in_loop_p99),
    ] {
        assert_positive(value, what);
    }
    assert!(
        added_ms_per_step.is_finite(),
        "the added-per-step figure must be finite, got {added_ms_per_step}"
    );
    println!(
        "decode window, SERIAL (one head call between decode steps on this thread): without \
         {tps_without:.2} tok/s, with {tps_with:.2} tok/s, added {added_ms_per_step:.3} ms per \
         step; in-loop ane mean/p50/p99 ms {in_loop_mean:.3}/{in_loop_p50:.3}/{in_loop_p99:.3}"
    );
    println!(
        "{}",
        LineFigures {
            mode: "decode",
            ane_p50_ms: Some(in_loop_p50),
            ane_p99_ms: Some(in_loop_p99),
            cpu_p50_ms: None,
            decode_tps_with: Some(tps_with),
            decode_tps_without: Some(tps_without),
        }
        .text()
    );
}
