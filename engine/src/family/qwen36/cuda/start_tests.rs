//! Unit tests of the CUDA serve start (CUDA-SERVE-1 tests 2b-5): the guard semantics of
//! KUBDAN-MSG-69 condition 2, the ignored-key start lines, the GGUF identity and the session
//! and reserve sizing. The text pins live in `start_pins.rs`; the health pins and the ignored
//! device test live in `start_health_tests.rs`.

use super::dims::{KvMode, StepDims};
use super::pinned::ROTARY_DIM;
use super::start::{
    BuildGuard, CUDA_ENGINE_BUILT, GGUF_BYTES, GGUF_NAME, already_built, check_gguf_identity,
    check_sha256, check_tiered_start, context_tokens, file_sha256, ignored_mlx_line,
    kv_bytes_per_position, kv_cache_bytes, kv_mode_of, load_options, pinned_gguf, serve_kv_mode,
    sha256_wanted, step_options, tiered_plan_of,
};
use super::state_layout::KvPool;
use crate::EngError;
use crate::family::QWEN3_6_35B_A3B;
use crate::family::gguf_model::Inventory;
use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::host::HostShape;
use crate::family::qwen36::settings::{
    Backend, KvDtype, Q36Env, Q36Options, Q36Settings, kv_dtype_name,
};
use std::path::Path;
use std::sync::{Mutex, atomic::Ordering};

/// The guard tests share the process-wide flag: one lock, the flag down at each end.
static GUARD_LOCK: Mutex<()> = Mutex::new(());

fn reset() {
    CUDA_ENGINE_BUILT.store(false, Ordering::Release);
}

fn refusal(result: Result<BuildGuard, EngError>) -> String {
    match result {
        Err(EngError::BadRequest(text)) => text,
        Err(other) => panic!("expected BadRequest, got {other:?}"),
        Ok(_) => panic!("expected the documented refusal, got Ok"),
    }
}

#[test]
fn the_second_cuda_build_is_refused_with_the_documented_text() {
    let _lock = GUARD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let first = BuildGuard::acquire().expect("the first build takes the flag");
    // The refusal is the packet's documented text (2.1), pinned as a LITERAL — comparing
    // `already_built()` with itself cannot fail (KUBDAN-MSG-72 item 5).
    let want = "engine config: the CUDA Q36 backend builds one engine per process: its \
                weights are leaked for the process lifetime and an in-process restart would \
                leak 20 GB again; configure a single qwen3_6_35b_a3b entry and restart the \
                process to reload";
    assert_eq!(already_built(), EngError::BadRequest(want.to_string()));
    assert_eq!(refusal(BuildGuard::acquire()), want);
    drop(first);
    assert!(
        BuildGuard::acquire().is_ok(),
        "an uncommitted drop must release"
    );
    reset();
}

#[test]
fn a_build_that_failed_before_the_leak_releases_the_guard() {
    let _lock = GUARD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    reset();
    {
        // A failure before the first leak drops the guard uncommitted on the way out.
        let _guard = BuildGuard::acquire().expect("the flag is down");
    }
    assert!(
        !CUDA_ENGINE_BUILT.load(Ordering::Acquire),
        "the flag stayed up"
    );
    assert!(BuildGuard::acquire().is_ok(), "the retry was refused");
    reset();
}

#[test]
fn a_failure_after_the_leak_keeps_the_guard_committed() {
    let _lock = GUARD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    reset();
    {
        let mut guard = BuildGuard::acquire().expect("the flag is down");
        guard.commit();
        // The leaks happened: a later step error unwinds `make` and drops the guard
        // COMMITTED (condition 2).
    }
    assert!(
        CUDA_ENGINE_BUILT.load(Ordering::Acquire),
        "the flag was released"
    );
    assert!(refusal(BuildGuard::acquire()).contains("one engine per process"));
    reset();
}

#[test]
fn ignored_mlx_keys_produce_a_start_line() {
    // The settings record the ignored keys; the start logs exactly one line per name.
    let options = Q36Options {
        cache_limit_bytes: Some(64),
        session_cap_bytes: Some(1),
        prefill_chunk: Some(8),
        ..Q36Options::default()
    };
    let got = Q36Settings::new(
        Path::new("/m"),
        Path::new("/t"),
        Some(&options),
        &Q36Env::default(),
        Backend::Cuda,
    )
    .unwrap();
    assert_eq!(
        got.ignored_mlx_keys,
        ["cache_limit_bytes", "session_cap_bytes", "prefill_chunk"]
    );
    for (key, line) in [
        (
            "cache_limit_bytes",
            "q36 cuda: cache_limit_bytes is an MLX allocator key and is ignored",
        ),
        (
            "session_cap_bytes",
            "q36 cuda: session_cap_bytes is an MLX session key and is ignored",
        ),
        (
            "prefill_chunk",
            "q36 cuda: prefill_chunk is an MLX chunked-prefill key and is ignored",
        ),
    ] {
        assert_eq!(ignored_mlx_line(key), line);
    }
    // The settings record only the three names; a foreign name still has its line (the pure
    // function is total, KUBDAN-MSG-71 item 5).
    assert_eq!(
        ignored_mlx_line("capture"),
        "q36 cuda: an unknown key of the object is ignored"
    );
}

#[test]
fn the_gguf_identity_is_checked_by_name_and_size() {
    let dir = tempfile::tempdir().expect("temp dir");
    let gguf = pinned_gguf(dir.path());
    assert_eq!(
        gguf.file_name().expect("name"),
        GGUF_NAME,
        "the name is pinned"
    );
    let text = |result: Result<(), EngError>| match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected Checkpoint, got {other:?}"),
    };
    let missing = text(check_gguf_identity(&gguf, false));
    assert!(
        missing.contains("is missing") && missing.contains(GGUF_NAME),
        "{missing}"
    );
    // A size off by one, then the pinned size: only the metadata is read (a sparse file, no
    // 20 GB written).
    let file = std::fs::File::create(&gguf).expect("create");
    file.set_len(GGUF_BYTES - 1).expect("truncate");
    let wrong = text(check_gguf_identity(&gguf, false));
    assert!(wrong.contains(&format!("{}", GGUF_BYTES - 1)), "{wrong}");
    assert!(wrong.contains(&GGUF_BYTES.to_string()), "{wrong}");
    file.set_len(GGUF_BYTES).expect("sparse to the pinned size");
    assert!(
        check_gguf_identity(&gguf, false).is_ok(),
        "metadata admits the size"
    );
    for (raw, want) in [(None, false), (Some("0"), false), (Some("1"), true)] {
        assert_eq!(sha256_wanted(raw), Ok(want));
    }
    // Every other spelling is a typed refusal naming the variable (KUBDAN-MSG-72 NIT): only
    // `1` turns the check on, `0` and unset leave it off, and `true`/`yes`/`` cannot promise
    // an unchecked check.
    for raw in ["yes", "true", ""] {
        let text = match sha256_wanted(Some(raw)) {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("raw {raw:?} was accepted: {other:?}"),
        };
        assert!(
            text.contains("Q36_VERIFY_GGUF_SHA256 must be 0 or 1"),
            "{text}"
        );
    }
}

#[test]
fn the_file_digest_and_the_mismatch_text_come_from_one_pass() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("tiny");
    std::fs::write(&path, b"abc").expect("write");
    let digest = file_sha256(&path).expect("digest");
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(
        check_sha256(&path, &digest).is_ok(),
        "a matching digest passes"
    );
    let wrong = match check_sha256(&path, &"0".repeat(64)) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected Checkpoint, got {other:?}"),
    };
    assert!(wrong.contains("does not match the pinned file"), "{wrong}");
}

#[test]
fn the_step_options_follow_the_context_limit_and_the_given_layout_and_mode() {
    // KUBDAN-MSG-72 item 1: step_options is pure in (limit, layout, kv_mode); make passes the
    // loaded weights' layout and the resolved mode, so the options carry whichever pair they
    // are given, unchanged.
    for layout in [WeightLayout::Tiled, WeightLayout::Native] {
        let options = step_options(32_768, layout, KvMode::Fp16).expect("the serve default");
        assert_eq!(options.context_tokens, 32_768);
        assert_eq!((options.delta_slots, options.forked_experts), (1, false));
        assert_eq!(
            options.tiered, None,
            "tiered is off unless a config key turns it on"
        );
        assert!(matches!(options.kv_mode, KvMode::Fp16));
        assert_eq!(options.layout, layout, "the layout passes through");
    }
    assert!(
        step_options(2, WeightLayout::Tiled, KvMode::Fp16).is_ok(),
        "the smallest limit the Mac arm accepts"
    );
    for limit in [0, 1] {
        let text = match step_options(limit, WeightLayout::Native, KvMode::Fp16) {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("limit {limit} was accepted: {other:?}"),
        };
        assert!(text.contains("context_limit must be at least 2"), "{text}");
        // start() validates through the same function before any device work.
        assert!(
            context_tokens(limit, KvMode::Fp16).is_err(),
            "start refuses limit {limit} too"
        );
    }
    // u32::MAX overflows no pure term (20,480 * 4,294,967,295 < 2^64): the largest limits are
    // refused by the load admission, which names the demand and the free figure.
    let huge =
        step_options(u32::MAX, WeightLayout::Tiled, KvMode::Fp16).expect("no pure term overflows");
    assert_eq!(
        context_tokens(u32::MAX, KvMode::Fp16),
        Ok(huge.context_tokens)
    );
    assert_eq!(
        kv_cache_bytes(huge.context_tokens, KvMode::Fp16),
        Ok(20_480 * 4_294_967_295u64)
    );
    assert!(
        kv_cache_bytes(usize::MAX, KvMode::Fp16).is_err(),
        "an overflowing product is refused"
    );
}

#[test]
fn the_kv_bytes_per_position_is_the_pool_figure_of_the_pinned_shape_in_both_modes() {
    // KUBDAN-MSG-72 item 6, carried to both modes (KV8-B B4): the admission's per-position
    // term is the SAME figure the state layout computes for the pinned shape — 20,480 B under
    // Fp16 and 10,880 B under Int8G32 — never an independent constant: the layout is the
    // source, the two literals are the pin.
    let dims = StepDims::new(
        &HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM).expect("shape"),
        &Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory"),
    )
    .expect("dims");
    for (mode, want) in [(KvMode::Fp16, 20_480u64), (KvMode::Int8G32, 10_880u64)] {
        let per_token = KvPool::bytes_per_token(&dims, mode).expect("bytes per token");
        assert_eq!(u64::try_from(per_token).expect("u64"), want, "{mode:?}");
        assert_eq!(
            kv_bytes_per_position(mode).expect("per position"),
            want,
            "{mode:?}"
        );
        // And the admission term is that figure per position.
        assert_eq!(
            kv_cache_bytes(1, mode).expect("one position"),
            want,
            "{mode:?}"
        );
    }
}

#[test]
fn the_kv_mode_of_the_config_maps_f32_to_fp16_and_q8_to_int8g32() {
    // Main's "f32" names the CONTROL cache, and on the CUDA arm the control cache is fp16.
    for layout in [WeightLayout::Tiled, WeightLayout::Native] {
        assert_eq!(
            kv_mode_of(settings(&KvDtype::F32).kv, layout).expect("f32"),
            KvMode::Fp16
        );
    }
    assert_eq!(
        kv_mode_of(settings(&KvDtype::Q8).kv, WeightLayout::Native).expect("q8 native"),
        KvMode::Int8G32
    );
    // The serve admission resolves the mode against the layout the load pins (Native), so a
    // q8 config serves Int8G32 and an absent or f32 config serves Fp16.
    assert_eq!(
        serve_kv_mode(&settings(&KvDtype::F32)).expect("absent or f32"),
        KvMode::Fp16
    );
    assert_eq!(
        serve_kv_mode(&settings(&KvDtype::Q8)).expect("q8"),
        KvMode::Int8G32
    );
}

#[test]
fn q8_with_the_tiled_layout_is_refused_by_the_mode_of_the_start() {
    // The typed refusal of the control path (the plan refuses it again, `check_layout_rows`):
    // the guard is the (Q8, Tiled) pair alone — both layouts pass with F32 and Native passes
    // with Q8 (the test above), so this run reaches its OWN guard.
    match kv_mode_of(settings(&KvDtype::Q8).kv, WeightLayout::Tiled) {
        Err(EngError::BadRequest(text)) => {
            assert!(text.contains("Int8G32") && text.contains("Tiled"), "{text}");
        }
        other => panic!("a Tiled q8 start must be refused: {other:?}"),
    }
    let kv = kv_cache_bytes(32_768, KvMode::Int8G32).expect("bytes");
    assert_eq!(
        kv,
        10_880 * 32_768,
        "the q8 admission term follows its mode"
    );
}

fn settings(kv: &KvDtype) -> Q36Settings {
    let entry = serde_json::from_str::<Q36Options>(&format!(
        "{{\"kv_dtype\": \"{}\"}}",
        kv_dtype_name(*kv)
    ))
    .expect("a valid qwen36 object");
    Q36Settings::new(
        std::path::Path::new("/m"),
        std::path::Path::new("/t"),
        Some(&entry),
        &Q36Env::default(),
        Backend::Cuda,
    )
    .expect("the CUDA settings")
}

#[test]
fn the_load_reserve_covers_the_kv_cache() {
    let kv = kv_cache_bytes(32_768, KvMode::Fp16).expect("bytes");
    assert_eq!(kv, 20_480 * 32_768);
    let options = load_options(kv);
    // work is 0: the workspace is spent before the admission reads free.
    assert_eq!(
        (
            options.reserve.kv_cache,
            options.reserve.headroom,
            options.reserve.work
        ),
        (671_088_640, 1 << 30, 0)
    );
    assert!(options.drop_page_cache.is_none());
    // MSG-70 condition 8: the load layout is written outright; since the S3c qualification
    // (KUBDAN-MSG-77) it is `Native`, and this one field of `load_options` is the switch.
    assert_eq!(options.layout, WeightLayout::Native);
}

fn cuda_settings(object: &str) -> Q36Settings {
    let options: Q36Options = serde_json::from_str(object).expect("a valid qwen36 object");
    Q36Settings::new(
        Path::new("/m"),
        Path::new("/t"),
        Some(&options),
        &Q36Env::default(),
        Backend::Cuda,
    )
    .expect("valid CUDA settings")
}

#[test]
fn start_without_the_tiered_key_builds_no_bank() {
    let plain = cuda_settings("{}");
    assert_eq!(
        tiered_plan_of(&plain),
        None,
        "tiered is off unless the member is set"
    );
    assert!(check_tiered_start(&plain).is_ok());
    let tiered = cuda_settings(
        r#"{"tiered": {"slots": 40, "threads": 4, "timeout_ms": 2500, "preload": true}}"#,
    );
    let plan = tiered_plan_of(&tiered).expect("the member asks for a plan");
    assert_eq!(
        (plan.slots_per_layer, plan.threads, plan.timeout_ms),
        (40, 4, 2500)
    );
    assert_eq!(
        (plan.swaps_milli, plan.decay_milli),
        (250, 999),
        "the measured defaults"
    );
    assert!(plan.preload, "the member asks for the preload");
    assert!(
        !tiered_plan_of(&cuda_settings(r#"{"tiered": {"slots": 40}}"#))
            .unwrap()
            .preload,
        "the banks start empty unless asked"
    );
    assert!(plan.check().is_ok());
    assert!(check_tiered_start(&tiered).is_ok());
}

#[test]
fn start_refuses_capture_with_tiered() {
    // The settings refuse the pair by name; the start checks it again on its own, so a settings
    // value built by hand cannot start a captured tiered engine.
    let mut tiered = cuda_settings(r#"{"tiered": {"slots": 40}}"#);
    tiered.capture = true;
    match check_tiered_start(&tiered) {
        Err(EngError::BadRequest(message)) => {
            assert!(
                message.contains("tiered and capture exclude each other"),
                "{message}"
            );
        }
        other => panic!("tiered with capture must be refused at the start: {other:?}"),
    }
}
