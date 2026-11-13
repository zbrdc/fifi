//! Golden capture harness for the chunked prefill: logits and exported-state digests of the
//! served numerics at named caps, one arm per process, so the promotion of the chunked prefill
//! can prove Qwen3.6's behavior byte-for-byte. Ignored: every arm loads the real pack (about
//! 21 GiB on the 3090, held exclusively, `--test-threads=1`).
//!
//! Arms, each its own process, every one captured twice in separate processes before a golden
//! is accepted:
//! - `golden_capture_exl3_fp16_arm`: `FIFI_GOLDEN_CAP` one of 128, 256, 512, 1024, 2048 on the
//!   EXL3 pack, fp16 KV, the session's own chunk partition (`feed_rows`).
//! - `golden_capture_exl3_kv8_arm`: `FIFI_GOLDEN_CAP` one of 16, 128, 256, 512, 1024, Int8G32
//!   with the fixed admission and one `feed_rows_chunked` call per cap-sized piece. At cap 16
//!   the admission builds the 128-row arena and the verdict reports `m_sess` 16; the arm asserts
//!   both figures.
//! - `golden_capture_gguf_fp16_arm`: the GGUF model dir of `Q36_CUDA_MODEL_DIR`,
//!   `FIFI_GOLDEN_CAP` 1024, fp16 — the one golden of the non-EXL3 arena path.
//! - `golden_capture_exl3_fp16_scribble_arm`: cap 1024 with `FIFI_GOLDEN_SCRIBBLE` set: the
//!   arena allocation is filled with 0xA5 before the zero fill, so a span the zero fill misses
//!   cannot hide behind fresh allocation state. Must equal the plain cap-1024 fp16 golden.
//!
//! The requested cap is asserted admitted immediately after `admit_prefill` (a narrowed rung
//! or an admission error fails the arm, writing a failure manifest with the verdict first), the
//! arena the session really built is asserted equal to the planned chunk table before anything
//! is fed, the stage rows are asserted equal to the partition the host golden pins, and every
//! arm's manifest records the environment, the pack hash, the GPU facts, the commit and the
//! stage digest. The GPU facts of the committed record (compute capability, SM count) come from
//! the device itself, not from the operator's JSON.
//!
//! Environment: `FIFI_GOLDEN_OUT` (required, artifacts root), `FIFI_GOLDEN_CAP` (required),
//! `FIFI_GOLDEN_COMMIT` and `FIFI_GOLDEN_PACK_SHA256` (required to update),
//! `FIFI_GOLDEN_GPU_JSON` (operator notes for the manifest only), `FIFI_GOLDEN_SCRIBBLE`
//! (the scribble arm only; any other arm aborts with it set), `FIFI_GOLDEN_UPDATE` (write the
//! goldens file instead of comparing), `FIFI_GOLDEN_REPLACE` (the one arm an update may
//! overwrite), plus the pack source of the arm (`FIFI_EXL3_CHECKPOINT` or `Q36_CUDA_MODEL_DIR`).
//! Any other `FIFI*`, `INFERENCE*` or `Q36*` variable aborts the arm: a golden must not depend
//! on an unrecorded knob.
//!
//! Artifacts land in `FIFI_GOLDEN_OUT/<run-id>/` (the run id carries the arm and a timestamp,
//! so no file is ever overwritten): the full logits row (`*.logits.f32`), the canonical export
//! (`*.state.bin`) and `*.manifest.json`. The committed goldens (`testdata/prefill-goldens.json`)
//! hold the digests and the provenance of each arm (pack hash, GPU, commit, prompt hash, cap,
//! kv, stage digest), not the bytes.

use std::path::{Path, PathBuf};

use serde_json::json;
use sha2::{Digest, Sha256};

use super::common_tests::dims;
use super::common_tests::inventory;
use super::dims::{KvMode, StepDims, StepOptions};
use super::exl3_load::load_exl3_dir;
use super::prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillArena, PrefillBudget, ZERO_CHUNK};
use super::prefill_exl3_step_device_tests::{SAMPLER_VOCAB, rules as step_rules};
use super::prefill_plan::{
    ChunkSlot, PrefillStep, chunk_list_with_depth_tier, chunk_tensors, exl3_chunk_tensors,
};
use super::resources::StepResources;
use super::state::export::ExportedState;
use super::step::ChunkVerdict;
use super::step::QwenResidentStep;
use crate::EngError;
use crate::family::gguf_model::open_qwen3_6_35b_a3b;
use crate::family::project::cuda::{
    CudaProjector, CudaWeights, LoadOptions, Reserve, WeightLayout,
};
use crate::family::session::TokenStepper;
use crate::kernels::cuda::{ArenaTensor, plan_arena};

/// The golden prompt: `FED_ROWS` rows are fed (the head) and the token after them goes through
/// decode, so the file needs `PROMPT_IDS` ids. `5,377 = 21 x 256 + 1`, past the 3,072 depth-tier
/// boundary, gives every arm its named shape: at caps 16, 128 and 256 the tail is a one-row
/// `Feed`; at caps 512, 1024 and 2048 it is a 257-row partial chunk; and at cap 2048 the chunk
/// at 4,096 (1,024 rows) is cut by the depth tier rather than the cap, while the 257-row tail
/// at 5,120 is cut by the prompt's end. The host golden test asserts these properties of the
/// same constant.
pub(super) const FED_ROWS: usize = 5_377;
/// The whole prompt: the fed head plus the one decode token.
pub(super) const PROMPT_IDS: usize = FED_ROWS + 1;

/// The golden prompt's ids, pinned into the binary: the file is committed testdata, and its
/// sha256 is asserted so a drifted fixture fails before the pack loads.
const IDS_JSON: &str = include_str!("testdata/prefill-golden-ids.json");
pub(super) const IDS_SHA256: &str =
    "b6b3989689d7014994c3e95b9da778930bb6bd06ac373bcb2be50981548aa7bc";

fn digest(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The prompt ids of the committed fixture.
pub(super) fn prompt_ids() -> Vec<u32> {
    let value: serde_json::Value = serde_json::from_str(IDS_JSON).expect("ids json");
    value["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("id") as u32)
        .collect()
}

/// The text form of a chunk table the arena build must match: numeric lines, one per tensor.
/// The host golden digests the same text (with the arena offsets beside it, host-only), and every
/// arm compares the table its session really built against the planned one through this text.
/// The slot names are explicit strings of this module, not `Debug` output, so a variant rename
/// cannot move a digest.
pub(super) fn table_text(list: &[(ChunkSlot, ArenaTensor)], total: usize, m: usize) -> String {
    let mut text = format!("m={m} total={total}\n");
    for (index, (slot, tensor)) in list.iter().enumerate() {
        text.push_str(&format!(
            "{index} {} bytes={} first={} last={}\n",
            slot_name(slot),
            tensor.bytes,
            tensor.first,
            tensor.last
        ));
    }
    text
}

/// The explicit name of a chunk slot, so table digests never hash `Debug` text.
fn slot_name(slot: &ChunkSlot) -> &'static str {
    match slot {
        ChunkSlot::Hidden => "hidden",
        ChunkSlot::IdsTok => "ids_tok",
        ChunkSlot::X => "x",
        ChunkSlot::X2 => "x2",
        ChunkSlot::XfA => "xf_a",
        ChunkSlot::XfF => "xf_f",
        ChunkSlot::Q8X => "q8_x",
        ChunkSlot::Q8X2 => "q8_x2",
        ChunkSlot::ProjD => "proj_d",
        ChunkSlot::ConvOut => "conv_out",
        ChunkSlot::OOut => "o_out",
        ChunkSlot::Q8OD => "q8_od",
        ChunkSlot::Q8OA => "q8_oa",
        ChunkSlot::ProjA => "proj_a",
        ChunkSlot::Q => "q",
        ChunkSlot::AttnOut => "attn_out",
        ChunkSlot::Delta => "delta",
        ChunkSlot::Logits => "logits",
        ChunkSlot::TopkId => "topk_id",
        ChunkSlot::TopkWeight => "topk_weight",
        ChunkSlot::SharedGate => "shared_gate",
        ChunkSlot::GuSh => "gu_sh",
        ChunkSlot::ActSh => "act_sh",
        ChunkSlot::Q8Ash => "q8_ash",
        ChunkSlot::ShOut => "sh_out",
        ChunkSlot::IdsG => "ids_g",
        ChunkSlot::IdsSrc1 => "ids_src1",
        ChunkSlot::IdsDst => "ids_dst",
        ChunkSlot::ExpertBounds => "expert_bounds",
        ChunkSlot::Q8S => "q8_s",
        ChunkSlot::Gu => "gu",
        ChunkSlot::Act => "act",
        ChunkSlot::Q8Act => "q8_act",
        ChunkSlot::Down => "down",
        ChunkSlot::Fixup => "fixup",
        ChunkSlot::Exl3Xh => "exl3_xh",
        ChunkSlot::Exl3Z => "exl3_z",
        ChunkSlot::Exl3Y => "exl3_y",
        ChunkSlot::Exl3Flag => "exl3_flag",
        ChunkSlot::DenseJobs => "dense_jobs",
        ChunkSlot::ExpertPerm => "expert_perm",
        ChunkSlot::ExpertXh => "expert_xh",
        ChunkSlot::ExpertZ => "expert_z",
    }
}

/// The explicit K/V mode string of the export digest: a `KvMode` variant rename must never move
/// an export digest.
fn kv_tag(mode: KvMode) -> &'static str {
    match mode {
        KvMode::Fp16 => "fp16",
        KvMode::Int8G32 => "int8g32",
    }
}

/// The planned chunk table of size `m` for `layout`, as the arena build must realize it. The
/// family (GGUF) table depends on the device's SM count, so the caller passes the real one.
pub(super) fn planned_table(dims: &StepDims, nsm: usize, m: usize, layout: WeightLayout) -> String {
    let list = match layout {
        WeightLayout::Exl3 => exl3_chunk_tensors(dims, m).expect("exl3 list"),
        _ => chunk_tensors(dims, m, nsm).expect("family list"),
    };
    let arena = plan_arena(&list.iter().map(|&(_, t)| t).collect::<Vec<_>>()).expect("arena");
    table_text(&list, arena.total, m)
}

fn goldens_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/family/qwen36/cuda/testdata/prefill-goldens.json")
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("FIFI_GOLDEN_OUT").expect("FIFI_GOLDEN_OUT"))
}

fn cap() -> usize {
    std::env::var("FIFI_GOLDEN_CAP")
        .expect("FIFI_GOLDEN_CAP")
        .parse()
        .expect("cap")
}

fn update() -> bool {
    std::env::var_os("FIFI_GOLDEN_UPDATE").is_some_and(|value| !value.is_empty())
}

/// Aborts unless every `FIFI*`, `INFERENCE*` and `Q36*` variable in the environment is one the
/// arm documents: a golden must not depend on an unrecorded knob. `FIFI_GOLDEN_SCRIBBLE` is
/// allowed only in the scribble arm, so a leaked variable cannot record a scribbled run under a
/// plain arm's name.
fn guard_env(scribble: bool) {
    let allowed: Vec<&str> = [
        "FIFI_GOLDEN_OUT",
        "FIFI_GOLDEN_UPDATE",
        "FIFI_GOLDEN_REPLACE",
        "FIFI_GOLDEN_COMMIT",
        "FIFI_GOLDEN_PACK_SHA256",
        "FIFI_GOLDEN_GPU_JSON",
        "FIFI_GOLDEN_CAP",
        "FIFI_EXL3_CHECKPOINT",
        "Q36_CUDA_MODEL_DIR",
    ]
    .into_iter()
    .chain(std::iter::once("FIFI_GOLDEN_SCRIBBLE").filter(|_| scribble))
    .collect();
    let stray: Vec<String> = std::env::vars()
        .filter(|(name, _)| {
            (name.starts_with("FIFI") || name.starts_with("INFERENCE") || name.starts_with("Q36"))
                && !allowed.contains(&name.as_str())
        })
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    assert!(
        stray.is_empty(),
        "the arm environment must be scrubbed; stray variables: {stray:?}"
    );
}

/// The provenance strings every update needs, validated before the pack loads: a non-empty
/// commit, a 64-hex-digit pack hash, a parseable operator GPU JSON, and a goldens file at the
/// baked path (the binary is built in the container and run on the host, so the baked path must
/// resolve where the arm runs).
fn validate_inputs() {
    let commit = std::env::var("FIFI_GOLDEN_COMMIT").unwrap_or_default();
    let pack = std::env::var("FIFI_GOLDEN_PACK_SHA256").unwrap_or_default();
    assert!(!commit.is_empty(), "FIFI_GOLDEN_COMMIT is required");
    assert!(
        pack.len() == 64 && pack.chars().all(|c| c.is_ascii_hexdigit()),
        "FIFI_GOLDEN_PACK_SHA256 must be a 64-hex-digit hash"
    );
    if let Ok(text) = std::env::var("FIFI_GOLDEN_GPU_JSON") {
        serde_json::from_str::<serde_json::Value>(&text).expect("FIFI_GOLDEN_GPU_JSON");
    }
    let path = goldens_path();
    assert!(
        path.is_file(),
        "the goldens file is missing at the baked path {}",
        path.display()
    );
}

/// The full-row logits of the last prompt token, f32 little-endian.
fn canonical_logits(row: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(row.len() * 4);
    for value in row {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The canonical export bytes: an explicit identity line, then every delta vector and every KV
/// plane in order, f32 as little-endian. The identity line is this module's own text (layout
/// tag, K/V mode string, token count) — never `Debug` output, so an enum or field rename cannot
/// move an export digest.
fn canonical_export(exported: &ExportedState) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(
        format!(
            "layout={} kv={} tokens={}\n",
            exported.identity.layout,
            kv_tag(exported.identity.kv_mode),
            exported.tokens
        )
        .as_bytes(),
    );
    for delta in &exported.delta {
        for value in delta.recurrent.iter().chain(delta.conv.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for layer in &exported.kv {
        for plane in &layer.planes {
            bytes.extend_from_slice(&(plane.len() as u64).to_le_bytes());
            bytes.extend_from_slice(plane);
        }
    }
    bytes
}

fn env_manifest() -> serde_json::Value {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(name, _)| {
            name.starts_with("FIFI") || name.starts_with("INFERENCE") || name.starts_with("Q36")
        })
        .collect();
    vars.sort();
    json!(vars)
}

/// What one arm captured: the logits row and export, the admission verdict text, the stage rows,
/// the digest of the arena table the session built, the GPU facts read from the device, and the
/// seconds of the admission plus feed.
struct Captured {
    logits: Vec<f32>,
    export: ExportedState,
    verdict: String,
    stages: Vec<(usize, usize, &'static str)>,
    table_sha256: String,
    gpu: serde_json::Value,
    seconds: f64,
}

/// Writes the failure manifest of an arm that did not admit at its requested cap (a narrowed
/// verdict or an admission error): the verdict text belongs in the artifacts even though the arm
/// fails.
fn write_failure_manifest(out_root: &Path, arm: &str, verdict_text: &str, kv8: bool) {
    let dir = out_root.join(format!(
        "{arm}-{}-fail",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("out dir");
    let manifest = json!({
        "arm": arm,
        "commit": std::env::var("FIFI_GOLDEN_COMMIT").unwrap_or_default(),
        "pack_sha256": std::env::var("FIFI_GOLDEN_PACK_SHA256").unwrap_or_default(),
        "cap": cap(),
        "kv": if kv8 { "int8g32" } else { "fp16" },
        "verdict": verdict_text,
        "env": env_manifest(),
    });
    std::fs::write(
        dir.join("failure.manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest"),
    )
    .expect("manifest file");
}

/// The arena table the session really built, in the shared text form.
fn built_table(arena: &PrefillArena) -> String {
    table_text(arena.list(), arena.arena().bytes(), arena.m_sess())
}

/// Asserts the verdict right after the admission, after writing the failure manifest of a
/// narrowed verdict.
fn assert_admitted(out_root: &Path, arm: &str, verdict: &ChunkVerdict, kv8: bool, wanted: usize) {
    let verdict_text = format!("{verdict:?}");
    let expected = if kv8 && wanted == 16 { 16 } else { wanted };
    if !matches!(verdict, ChunkVerdict::Ready { m_sess } if *m_sess == expected) {
        write_failure_manifest(out_root, arm, &verdict_text, kv8);
    }
    assert!(
        matches!(verdict, ChunkVerdict::Ready { m_sess } if *m_sess == expected),
        "the arm must be admitted at the requested cap {wanted}: {verdict_text}"
    );
}

/// Loads the arm's weights, runs the chunked prefill of the golden prompt at the explicit cap,
/// asserting the admission, the built arena and the stage rows along the way. `scribble` is the
/// arm's identity, not an environment read: the plain arms pass `false` so a leaked
/// `FIFI_GOLDEN_SCRIBBLE` cannot scribble or gate anything.
fn capture(
    arm: &str,
    kv8: bool,
    layout: WeightLayout,
    scribble: bool,
) -> Result<Captured, EngError> {
    guard_env(scribble);
    validate_inputs();
    // Read FIFI_GOLDEN_OUT before the pack loads: a missing value must fail here, not after.
    let out_root = out_dir();

    let mut hasher = Sha256::new();
    hasher.update(IDS_JSON.as_bytes());
    assert_eq!(
        format!("{:x}", hasher.finalize()),
        IDS_SHA256,
        "testdata/prefill-golden-ids.json drifted; update the sha or restore the fixture"
    );
    let ids = prompt_ids();
    assert!(
        ids.len() >= PROMPT_IDS,
        "the ids file needs {PROMPT_IDS} ids"
    );
    let head: Vec<u32> = ids.iter().take(FED_ROWS).copied().collect();
    let last = ids[FED_ROWS];

    let (projector, weights) = match layout {
        WeightLayout::Exl3 => {
            let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
            let projector = CudaProjector::new()?;
            let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default())?;
            (projector, weights)
        }
        WeightLayout::Native => {
            let dir = crate::test_paths::required("Q36_CUDA_MODEL_DIR");
            let gguf_path = Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf");
            let model = open_qwen3_6_35b_a3b(&gguf_path, None)?;
            let projector = CudaProjector::new()?;
            let device = projector
                .load_model(
                    &model,
                    LoadOptions {
                        reserve: Reserve::default(),
                        drop_page_cache: None,
                        layout: WeightLayout::Native,
                    },
                )?
                .device;
            (projector, device)
        }
        other => panic!("the golden arms cover Exl3 and Native, not {other:?}"),
    };
    let free = projector.memory_info()?.free;
    // The GPU facts of the record come from the device itself, not the operator's JSON.
    let facts = projector.context().mmq_device_facts()?;
    let gpu = json!({ "cc": facts.cc, "nsm": facts.nsm });

    let opts = StepOptions {
        kv_mode: if kv8 { KvMode::Int8G32 } else { KvMode::Fp16 },
        context_tokens: 8_192,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    };
    let res: &'static StepResources = Box::leak(Box::new(StepResources::new(
        projector.context(),
        dims(),
        &inventory(),
        &opts,
    )?));
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let mut step = QwenResidentStep::new(projector.context(), res, weights, &opts, step_rules())?;

    let wanted = cap();
    let began = std::time::Instant::now();
    // An admission error also writes its failure manifest: the reason belongs in the artifacts
    // as much as a narrowed verdict does.
    let admit =
        |step: &mut QwenResidentStep, budget: PrefillBudget, gate: bool, kv8_gate: bool| match step
            .admit_prefill(budget, gate, kv8_gate, false)
        {
            Ok(verdict) => verdict,
            Err(error) => {
                write_failure_manifest(&out_root, arm, &format!("admit error: {error:?}"), kv8);
                panic!("{arm}: admit_prefill failed: {error:?}");
            }
        };
    let verdict = if kv8 {
        step.set_kv8_chunked_prefill(true)?;
        step.set_m_sess_cap(wanted as u32);
        let verdict = admit(
            &mut step,
            PrefillBudget {
                device_free: free.saturating_sub(FIXED_ADMISSION_MARGIN),
                extras: Vec::new(),
            },
            true,
            true,
        );
        assert_admitted(&out_root, arm, &verdict, kv8, wanted);
        verdict
    } else {
        step.set_m_sess_cap(wanted as u32);
        let verdict = admit(
            &mut step,
            PrefillBudget {
                device_free: free.saturating_sub(1 << 30),
                extras: Vec::new(),
            },
            true,
            false,
        );
        assert_admitted(&out_root, arm, &verdict, kv8, wanted);
        verdict
    };
    let verdict_text = format!("{verdict:?}");
    println!("admission: {verdict_text}");

    // The verdict and the arena must agree, at the figures the KV8 cap-16 path defines: the
    // verdict reports the 16-row dispatch while the arena is the 128-row one.
    let expected_verdict = if kv8 && wanted == 16 { 16 } else { wanted };
    let expected_arena = if kv8 && wanted == 16 { 128 } else { wanted };
    let arena = step
        .test_prefill_arena()
        .expect("a Ready verdict implies a built arena");
    assert_eq!(arena.m_sess(), expected_arena, "{verdict_text}");
    assert!(
        matches!(&verdict, ChunkVerdict::Ready { m_sess } if *m_sess == expected_verdict),
        "the arm must be admitted at the requested cap {wanted}: {verdict_text}"
    );

    // The built arena table equals the planned one, asserted before anything is fed: the golden
    // cannot stay green over a session that never builds the table the plan digests. The family
    // (GGUF) table depends on the real SM count; the EXL3 table does not.
    let built = built_table(arena);
    let planned = planned_table(&dims(), facts.nsm, expected_arena, layout);
    assert_eq!(
        built, planned,
        "the built arena table must equal the planned one"
    );
    let table_sha256 = digest(&built);

    // The scribble arm's fill really ran: the whole fresh chunk arena in ZERO_CHUNK pieces.
    if scribble {
        let writes = arena.arena().bytes().div_ceil(ZERO_CHUNK);
        let scribbles =
            super::prefill_driver::io::SCRIBBLES.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            scribbles, writes,
            "the scribble fill must cover the whole arena ({writes} pieces)"
        );
    }

    if kv8 {
        for piece in head.chunks(wanted) {
            step.feed_rows_chunked(piece, wanted)?;
        }
    } else {
        step.feed_rows(&head)?;
    }

    let logits = step.logits(last)?;
    let seconds = began.elapsed().as_secs_f64();
    assert_eq!(step.tokens(), PROMPT_IDS, "every prompt row is committed");
    let stages: Vec<_> = step
        .last_prefill_report()
        .steps()
        .iter()
        .map(|record| {
            let kind = match record.kind {
                super::run_report::StepKind::Chunk => "chunk",
                super::run_report::StepKind::Feed => "feed",
            };
            (record.p0, record.rows, kind)
        })
        .collect();
    let expected: Vec<(usize, usize, &'static str)> =
        chunk_list_with_depth_tier(0, FED_ROWS, wanted, false)
            .expect("chunk list")
            .iter()
            .map(|step| match step {
                PrefillStep::Chunk { p0, rows } => (*p0, *rows, "chunk"),
                PrefillStep::Feed { p0 } => (*p0, 1, "feed"),
            })
            .collect();
    assert_eq!(
        stages, expected,
        "the stage rows must be the partition the host golden pins"
    );
    let export = step.export_state()?;
    Ok(Captured {
        logits,
        export,
        verdict: verdict_text,
        stages,
        table_sha256,
        gpu,
        seconds,
    })
}

/// Reads the goldens file; the committed file is the only source (an absent file means the
/// binary is running from the wrong tree, not a fresh start).
fn load_goldens(path: &Path) -> serde_json::Value {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).expect("goldens json"),
        Err(error) => panic!("{}: {error}", path.display()),
    }
}

/// Writes the goldens file atomically with sorted keys at every level, a trailing newline, and a
/// process-unique temporary name (the host and device tests share it).
pub(super) fn store_goldens(path: &Path, goldens: serde_json::Value) {
    fn sorted(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().cloned().collect();
                keys.sort();
                let mut out = serde_json::Map::new();
                for key in keys {
                    let value = sorted(map[&key].clone());
                    out.insert(key, value);
                }
                serde_json::Value::Object(out)
            }
            other => other,
        }
    }
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(&sorted(goldens)).expect("goldens");
    bytes.push(b'\n');
    std::fs::write(&tmp, bytes).expect("goldens write");
    std::fs::rename(&tmp, path).expect("goldens rename");
}

/// Runs one arm end to end: capture, artifacts, digest, golden compare or update. A scribble arm
/// must equal the plain arm's pinned golden, proving the zero fill covers what the kernels read.
fn run_arm(arm: &str, plain_twin: Option<&str>, kv8: bool, layout: WeightLayout, scribble: bool) {
    let captured = capture(arm, kv8, layout, scribble).expect("capture");
    let Captured {
        logits,
        export,
        verdict,
        stages,
        table_sha256,
        gpu,
        seconds,
    } = captured;
    assert!(logits.len() >= SAMPLER_VOCAB, "a full logits row");

    let run_id = format!(
        "{arm}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0)
    );
    let dir = out_dir().join(&run_id);
    std::fs::create_dir_all(&dir).expect("out dir");
    let logits_bytes = canonical_logits(&logits);
    let export_bytes = canonical_export(&export);
    std::fs::write(dir.join("logits.f32"), &logits_bytes).expect("logits file");
    std::fs::write(dir.join("state.bin"), &export_bytes).expect("state file");
    let logits_sha = format!("{:x}", Sha256::digest(&logits_bytes));
    let export_sha = format!("{:x}", Sha256::digest(&export_bytes));
    let stages_text = stages
        .iter()
        .map(|(p0, rows, kind)| format!("{kind} p0={p0} rows={rows}\n"))
        .collect::<String>();
    let stage_sha = digest(&stages_text);

    let commit = std::env::var("FIFI_GOLDEN_COMMIT").unwrap_or_default();
    let pack = std::env::var("FIFI_GOLDEN_PACK_SHA256").unwrap_or_default();
    let manifest = json!({
        "arm": arm,
        "commit": commit,
        "pack_sha256": pack,
        "gpu": {
            "cc": gpu["cc"],
            "nsm": gpu["nsm"],
            "notes": serde_json::from_str::<serde_json::Value>(
                &std::env::var("FIFI_GOLDEN_GPU_JSON").unwrap_or_else(|_| "{}".into()),
            ).expect("gpu json"),
        },
        "prompt": {
            "sha256": IDS_SHA256,
            "ids": prompt_ids().len(),
            "fed_rows": FED_ROWS,
            "prompt_ids": PROMPT_IDS,
        },
        "kv": if kv8 { "int8g32" } else { "fp16" },
        "cap": cap(),
        "verdict": verdict,
        "stages": stages,
        "stage_sha256": stage_sha,
        "table_sha256": table_sha256,
        "gpu": gpu,
        "logits_sha256": logits_sha,
        "export_sha256": export_sha,
        "seconds": seconds,
        "env": env_manifest(),
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest"),
    )
    .expect("manifest file");
    println!("PASS {arm}: logits {logits_sha}, export {export_sha}, {seconds:.1} s");

    let record = json!({
        "logits_sha256": logits_sha,
        "export_sha256": export_sha,
        "stage_sha256": stage_sha,
        "verdict": verdict,
        "commit": commit,
        "pack_sha256": pack,
        "prompt_sha256": IDS_SHA256,
        "gpu_cc": gpu["cc"],
        "gpu_nsm": gpu["nsm"],
        "cap": cap(),
        "kv": if kv8 { "int8g32" } else { "fp16" },
    });
    let path = goldens_path();
    let mut goldens = load_goldens(&path);
    if update() {
        let replace = std::env::var("FIFI_GOLDEN_REPLACE").is_ok_and(|value| value == arm);
        let existing = goldens["arms"].get(arm);
        if let Some(existing) = existing.filter(|old| *old != &record) {
            assert!(
                replace,
                "{arm} is pinned differently; old:\n{existing}\nnew:\n{record}\n\
                 set FIFI_GOLDEN_REPLACE={arm} to overwrite"
            );
        }
        if let Some(twin) = plain_twin
            && let Some(pinned) = goldens["arms"].get(twin)
            && *pinned != record
        {
            panic!("{arm} must equal its plain twin {twin}; twin:\n{pinned}\nthis arm:\n{record}");
        }
        goldens["arms"]
            .as_object_mut()
            .expect("arms object")
            .insert(arm.into(), record);
        store_goldens(&path, goldens);
        println!("UPDATED {arm} in {}", path.display());
    } else {
        let arms = goldens["arms"].as_object().expect("arms object");
        let pinned = arms
            .get(arm)
            .unwrap_or_else(|| panic!("no golden for {arm}; capture with FIFI_GOLDEN_UPDATE=1"));
        assert_eq!(
            pinned["logits_sha256"].as_str(),
            Some(logits_sha.as_str()),
            "{arm}: logits digest moved"
        );
        assert_eq!(
            pinned["export_sha256"].as_str(),
            Some(export_sha.as_str()),
            "{arm}: export digest moved"
        );
        assert_eq!(
            pinned["stage_sha256"].as_str(),
            Some(stage_sha.as_str()),
            "{arm}: stage rows moved"
        );
        if let Some(twin) = plain_twin {
            let pinned = arms
                .get(twin)
                .unwrap_or_else(|| panic!("no golden for the plain twin {twin}"));
            assert_eq!(
                pinned["logits_sha256"].as_str(),
                Some(logits_sha.as_str()),
                "{arm}: the scribble arm must equal {twin}"
            );
            assert_eq!(
                pinned["export_sha256"].as_str(),
                Some(export_sha.as_str()),
                "{arm}: the scribble arm must equal {twin}"
            );
            println!("SCRIBBLE MATCH {twin}");
        }
        println!("GOLDEN MATCH {arm}");
    }
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): the real pack in FIFI_EXL3_CHECKPOINT"]
fn golden_capture_exl3_fp16_arm() {
    run_arm(
        &format!("exl3-fp16-{}", cap()),
        None,
        false,
        WeightLayout::Exl3,
        false,
    );
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): the real pack in FIFI_EXL3_CHECKPOINT, Int8G32"]
fn golden_capture_exl3_kv8_arm() {
    run_arm(
        &format!("exl3-kv8-{}", cap()),
        None,
        true,
        WeightLayout::Exl3,
        false,
    );
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): the GGUF model in Q36_CUDA_MODEL_DIR"]
fn golden_capture_gguf_fp16_arm() {
    run_arm(
        &format!("gguf-fp16-{}", cap()),
        None,
        false,
        WeightLayout::Native,
        false,
    );
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): the real pack in FIFI_EXL3_CHECKPOINT, scribbled arena"]
fn golden_capture_exl3_fp16_scribble_arm() {
    assert!(
        std::env::var_os("FIFI_GOLDEN_SCRIBBLE").is_some(),
        "this arm runs with FIFI_GOLDEN_SCRIBBLE set"
    );
    assert_eq!(cap(), 1024, "the scribble arm is the plain cap-1024 twin");
    run_arm(
        "exl3-fp16-1024-scribble",
        Some("exl3-fp16-1024"),
        false,
        WeightLayout::Exl3,
        true,
    );
}
