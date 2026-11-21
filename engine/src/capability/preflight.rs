//! The compatibility verdict for one model directory on this machine,
//! computed before anything is loaded.
//!
//! Qwen3.6 gets the deep path because its validators are reusable as-is:
//! option checking is `Q36Settings::new` (the exact function the factory
//! runs at start), the tokenizer verdict is `TokenizerBundle::load` with
//! the family's own pin, and the byte estimate uses
//! `settings::cache_growth_per_position` and `settings::delta_state_bytes`
//! — the authorities the session admission itself charges by. Other
//! servable families get an honest shallow path (inventory, tokenizer
//! presence, no invented estimate) until their validators are composed
//! the same way.

use serde::Serialize;
use std::path::{Path, PathBuf};

use super::{Probe, inspect::ModelInspection, probe::HardwareFacts};
use crate::memory::MemoryPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Supported, and the estimated plan fits the probed budget.
    Ready,
    /// Supported and fits, with warnings the user should read first.
    ReadyWithAdjustments,
    /// Supported, but the estimated plan exceeds the probed budget.
    ResourceConstrained,
    /// Implemented but not qualified end to end for this combination.
    Experimental,
    /// This build cannot serve this model directory.
    Unsupported,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Ready => "ready",
            Verdict::ReadyWithAdjustments => "ready with adjustments",
            Verdict::ResourceConstrained => "resource-constrained",
            Verdict::Experimental => "experimental",
            Verdict::Unsupported => "unsupported",
        }
    }

    /// Whether serving should be attempted with these settings. The CLI
    /// exit code follows this; nothing is auto-adjusted either way.
    pub fn servable(self) -> bool {
        matches!(
            self,
            Verdict::Ready | Verdict::ReadyWithAdjustments | Verdict::Experimental
        )
    }
}

/// The three independent facts behind the summary verdict. A single
/// verdict alone conflates them: a model can be *compatible* but
/// *unqualified*, or qualified but *not fitting* this machine right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compatibility {
    Supported,
    Unsupported,
    /// Inspection was not deep enough to decide (shallow path).
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qualification {
    /// Measured and gated for this combination (see doc/explanation/status.md).
    Qualified,
    /// Implemented, but the quality gate for this combination has not passed.
    Experimental,
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceFit {
    Fits,
    DoesNotFit,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Dimensions {
    pub compatibility: Compatibility,
    pub qualification: Qualification,
    pub resource_fit: ResourceFit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
    Info,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

impl Check {
    fn new(name: &str, status: CheckStatus, detail: String) -> Self {
        Self {
            name: name.to_string(),
            status,
            detail,
        }
    }
}

/// The memory estimate: checkpoint **file** bytes plus closed-form cache
/// and state bytes. File bytes are not promised to equal runtime-resident
/// weight allocations — residency is each loader's decision (the CUDA
/// GGUF arm uploads the pinned file's tensors; tiered and host-offload
/// modes deliberately do not) — so the weight term is an upper-bound
/// planning figure, labeled as file bytes wherever rendered. Workspace, activations, and driver overhead are **not**
/// in `total_bytes`; on the CUDA path they enter through the runtime
/// admission charges ([`Estimate::admission_charges_bytes`], modeled on the
/// load-time ledger), and load-time admission remains the authority.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Estimate {
    pub weight_bytes: u64,
    pub kv_bytes_per_position: u64,
    pub context_limit: u32,
    pub kv_bytes: u64,
    pub fixed_state_bytes: u64,
    pub total_bytes: u64,
    /// What the estimate was compared against, in the budget's own words
    /// (for example "free VRAM of device 0, measured now").
    pub budget_kind: String,
    pub budget_free_bytes: Option<u64>,
    /// The admission ceiling the demand was compared against: the measured
    /// free bytes on the CUDA path (the margin is inside the demand), the
    /// `MemoryPolicy` share of them on the MLX path.
    pub budget_limit_bytes: Option<u64>,
    /// Whether `budget_limit_bytes` is the `MemoryPolicy` share of the free
    /// bytes (MLX) rather than the free bytes themselves (CUDA).
    pub budget_limit_is_policy_share: bool,
    /// The largest context whose estimate fits the budget, when that is
    /// smaller than the requested one. A proposal, never applied.
    pub largest_fitting_context: Option<u32>,
    /// The runtime admission demand charged after the weights on the CUDA
    /// path: cache, fixed state, prefix device tip, staged KV8 rows, part
    /// charge, spare minimum, admission margin, and the measured runtime
    /// overhead floor. Zero on the MLX path, which the runtime budgets by
    /// the memory policy share instead.
    pub admission_charges_bytes: u64,
    /// The named terms behind `admission_charges_bytes`, for the rendered
    /// breakdown. Empty when the estimate has no charge model.
    pub admission_charge_terms: Vec<(String, u64)>,
    /// The estimate's own fit call against the measured budget.
    pub fit: ResourceFit,
    /// `true` when `fit` is `Unknown` because the demand lands inside the
    /// measurement band around the budget boundary (±`ADMISSION_UNCERTAINTY_BAND_BYTES`):
    /// load-time admission decides, not the estimate.
    pub borderline: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreflightReport {
    pub model_dir: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub family: Option<String>,
    pub verdict: Verdict,
    pub dimensions: Dimensions,
    pub checks: Vec<Check>,
    pub estimate: Option<Estimate>,
    /// The planned decisions behind the estimate: what each choice landed
    /// on, the alternatives the plan weighed, and whether the choice is
    /// fixed at engine start or re-decided at load. Flint EXL3 fills these
    /// from the same pure functions its loader runs; Qwen3.6 EXL3 from its
    /// pack walk; empty for estimates that compose no decisions.
    pub decisions: Vec<PlanDecision>,
    /// The single most useful next action, in one sentence.
    pub next_step: String,
}

/// The candidate settings a preflight evaluates. `None` fields take the
/// family's own resolved defaults, exactly as a config file without the
/// key would.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidate {
    pub family: Option<String>,
    pub context_limit: Option<u32>,
    pub kv_dtype: Option<String>,
    /// The candidate's chunked-prefill width, when it names one; unset resolves
    /// to the family's default exactly as at startup. Read only by families
    /// whose settings resolve a prefill width.
    pub prefill_rows: Option<u32>,
}

pub use crate::family::plan_decision::{PlanAlternative, PlanDecision, PlanFixity};

/// Run a preflight. `facts` is probed by the caller so one probe can
/// serve a whole report. Inspection failures (not a directory, unreadable)
/// are the only `Err`: everything about a readable directory, however
/// wrong for serving, comes back as a verdict with reasons.
pub fn preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    candidate: &Candidate,
    facts: &HardwareFacts,
) -> Result<PreflightReport, crate::EngError> {
    let inspection = super::inspect::inspect(model_dir, tokenizer_dir)?;
    Ok(evaluate(
        model_dir,
        tokenizer_dir,
        &inspection,
        candidate,
        facts,
    ))
}

fn evaluate(
    model_dir: &Path,
    tokenizer_dir: &Path,
    inspection: &ModelInspection,
    candidate: &Candidate,
    facts: &HardwareFacts,
) -> PreflightReport {
    let mut checks = Vec::new();
    inventory_checks(inspection, &mut checks);

    use crate::family::inkling2_mini;
    if candidate
        .family
        .as_deref()
        .is_none_or(|family| family == inkling2_mini::FAMILY)
        && inspection
            .gguf_files
            .iter()
            .any(|file| file.inkling2_loop.is_some())
    {
        for file in &inspection.gguf_files {
            let Some(geometry) = &file.inkling2_loop else {
                continue;
            };
            let (status, detail) = match geometry {
                Ok(g) => (
                    CheckStatus::Pass,
                    format!(
                        "single-file checkpoint, {} physical layers, {} layer visits ({} global, {} sliding-window), loop {}..{} x{}, hidden {}, {} of {} experts, context {}, window {}",
                        g.physical_layers,
                        g.logical_layers,
                        g.global_layers,
                        g.local_layers,
                        g.core_start,
                        g.core_end,
                        g.passes,
                        g.hidden_size,
                        g.experts_per_token,
                        g.experts,
                        g.context_length,
                        g.sliding_window,
                    ),
                ),
                Err(error) => (CheckStatus::Fail, error.clone()),
            };
            checks.push(Check::new(
                "Inkling 2 Mini GGUF layout",
                status,
                format!("{}: {detail}", file.file),
            ));
        }
        return inkling2_mini_report(
            model_dir,
            tokenizer_dir,
            checks,
            Vec::new(),
            "Inkling 2 Mini GGUF serving is unsupported",
            "Use the GGUF header inventory to validate the checkpoint layout.",
        );
    }
    if candidate.family.as_deref() == Some(inkling2_mini::FAMILY)
        || (candidate.family.is_none()
            && inspection
                .config_model_types
                .iter()
                .any(|kind| kind == inkling2_mini::MODEL_TYPE))
    {
        return inkling2_mini_preflight(model_dir, tokenizer_dir, candidate, checks);
    }

    let family = identify_family(inspection, candidate, &mut checks);
    let family_name = family.as_deref();
    if family_name == Some(crate::family::qwen36::factory::FAMILY) {
        // `None` when this binary has no GPU backend at all;
        // `backend_of_build`'s fallback arm must not masquerade as one.
        let backend =
            (!facts.backends.none()).then(crate::family::qwen36::factory::backend_of_build);
        return q36_preflight(
            model_dir,
            tokenizer_dir,
            inspection,
            candidate,
            facts,
            backend,
            checks,
        );
    }
    if family_name == Some(crate::family::flash_next::factory::FAMILY) {
        return rows_or_shallow(
            model_dir,
            tokenizer_dir,
            inspection,
            family,
            candidate,
            facts,
            checks,
            crate::family::flash_next::preflight_rows_entry(),
        );
    }
    if family_name == Some(crate::family::flint::FAMILY) {
        return flint_preflight(
            model_dir,
            tokenizer_dir,
            inspection,
            candidate,
            facts,
            flint_cuda_backend(),
            checks,
        );
    }
    shallow_preflight(model_dir, tokenizer_dir, inspection, family, checks)
}

/// Whether this build can serve Flint on CUDA at all. The estimate's
/// deep branch needs the same build; tests inject `true`.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn flint_cuda_backend() -> bool {
    crate::family::flint::factory::check_backend().is_ok()
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn flint_cuda_backend() -> bool {
    false
}

// Facts in, rows out: one function per family behind the dispatch, so the
// report grows by a dispatch line, not a hand-written arm per family. The
// shape itself lives with the shared trait, in family::plan_rows.
use crate::family::plan_rows::{RowsFit, RowsRequest, RowsSource};

/// The report every deep arm finishes with: the shared dimensions and
/// verdict over the arm's checks, qualification and decisions. `estimate`
/// is what the arm measured and the report stores; `fit_estimate`, when a
/// rows family restates its context fit as the fields the shared verdict
/// reads, feeds only the verdict and dimensions and is never stored.
#[allow(clippy::too_many_arguments)] // the report's identity, checks, estimate, rows and qualification are one composition, not a config object
fn finish_report(
    model_dir: &Path,
    tokenizer_dir: &Path,
    family: Option<String>,
    checks: Vec<Check>,
    estimate: Option<&Estimate>,
    fit_estimate: Option<&Estimate>,
    decisions: Vec<PlanDecision>,
    qualification: Qualification,
) -> PreflightReport {
    let fit = fit_estimate.or(estimate);
    let dimensions = q36_dimensions(&checks, fit, qualification);
    let (verdict, next_step) = q36_verdict(&checks, fit, qualification);
    PreflightReport {
        model_dir: model_dir.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        family,
        verdict,
        dimensions,
        checks,
        decisions,
        estimate: estimate.cloned(),
        next_step,
    }
}

/// The report a `RowsSource` family gets: the source's rows as the
/// decisions, its refusal as one failed check in the loader's own voice,
/// a fact it could not probe as a warning (not the pack's fault), and the
/// configured context's fit as data, a constrained verdict with a warning,
/// not a pass. No byte estimate is composed for these families; load-time
/// admission stays authoritative.
#[allow(clippy::too_many_arguments)] // forwarding the arm's inputs; the family grouping lives in the source, not here
fn rows_arm(
    model_dir: &Path,
    tokenizer_dir: &Path,
    family: Option<String>,
    source: RowsSource,
    candidate: &Candidate,
    facts: &HardwareFacts,
    qualification: Qualification,
    mut checks: Vec<Check>,
) -> PreflightReport {
    let (rows, note, fit) = match source(&RowsRequest {
        model_dir,
        tokenizer_dir,
        context: candidate.context_limit,
        kv_dtype: candidate.kv_dtype.as_deref(),
        prefill_rows: candidate.prefill_rows,
        facts,
    }) {
        Ok(answer) => answer,
        Err(error) => {
            checks.push(Check::new(
                "load plan",
                CheckStatus::Fail,
                format!(
                    "the load would refuse this pack: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish_report(
                model_dir,
                tokenizer_dir,
                family,
                checks,
                None,
                None,
                Vec::new(),
                qualification,
            );
        }
    };
    let (status, detail, constrained) = match fit {
        // Only the fields the shared verdict and dimensions read; the rest
        // stay empty because no byte estimate was measured.
        RowsFit::Constrained { requested, largest } => (
            CheckStatus::Warn,
            format!(
                "the requested {requested} positions do not fit; the largest fitting context is {largest}, a multiple of 1024 — a proposal, never applied; {note}"
            ),
            Some(Estimate {
                weight_bytes: 0,
                kv_bytes_per_position: 0,
                context_limit: requested,
                kv_bytes: 0,
                fixed_state_bytes: 0,
                total_bytes: 0,
                budget_kind: String::new(),
                budget_free_bytes: None,
                budget_limit_bytes: None,
                budget_limit_is_policy_share: false,
                largest_fitting_context: Some(largest),
                admission_charges_bytes: 0,
                admission_charge_terms: Vec::new(),
                fit: ResourceFit::DoesNotFit,
                borderline: false,
            }),
        ),
        RowsFit::Fits if !rows.is_empty() => (
            CheckStatus::Pass,
            format!(
                "{} decision rows from the same open, walk and plan the loader runs; {note}",
                rows.len()
            ),
            None,
        ),
        // A grammar that answers Fits carries rows; Unknown never does.
        // The defensive arm keeps a stray empty answer a warning, not a
        // pass.
        RowsFit::Unknown | RowsFit::Fits => {
            (CheckStatus::Warn, format!("no decision rows: {note}"), None)
        }
    };
    checks.push(Check::new("load plan", status, detail));
    finish_report(
        model_dir,
        tokenizer_dir,
        family,
        checks,
        None,
        constrained.as_ref(),
        rows,
        qualification,
    )
}

/// The rows-or-shallow dispatch: the rows arm when this build carries the
/// family's entry, the shallow report when it does not — a build without
/// the load path must not call the family unsupported. The qualification
/// comes from this caller, not the arm.
#[allow(clippy::too_many_arguments)] // a router over the arm's inputs plus the build's entry
fn rows_or_shallow(
    model_dir: &Path,
    tokenizer_dir: &Path,
    inspection: &ModelInspection,
    family: Option<String>,
    candidate: &Candidate,
    facts: &HardwareFacts,
    checks: Vec<Check>,
    entry: Option<RowsSource>,
) -> PreflightReport {
    match entry {
        Some(source) => rows_arm(
            model_dir,
            tokenizer_dir,
            family,
            source,
            candidate,
            facts,
            Qualification::Experimental,
            checks,
        ),
        None => shallow_preflight(model_dir, tokenizer_dir, inspection, family, checks),
    }
}

fn inventory_checks(inspection: &ModelInspection, checks: &mut Vec<Check>) {
    for fact in &inspection.gguf_files {
        checks.push(Check::new(
            "gguf header",
            CheckStatus::Pass,
            format!(
                "{}: {} bytes, {} tensors{}",
                fact.file,
                fact.file_bytes,
                fact.tensor_count,
                fact.architecture
                    .as_deref()
                    .map_or(String::new(), |arch| format!(", architecture {arch}")),
            ),
        ));
    }
    for error in &inspection.gguf_errors {
        checks.push(Check::new("gguf header", CheckStatus::Fail, error.clone()));
    }
    if inspection.safetensors_files > 0 {
        checks.push(Check::new(
            "safetensors shards",
            CheckStatus::Info,
            format!(
                "{} files, {} bytes (contents are the loader's check, not inspected here)",
                inspection.safetensors_files, inspection.safetensors_bytes
            ),
        ));
    }
    if inspection.weight_bytes() == 0 {
        checks.push(Check::new(
            "weights",
            CheckStatus::Fail,
            "no .gguf or .safetensors weight files in the directory".to_string(),
        ));
    }
    if inspection.tokenizer_missing.is_empty() {
        checks.push(Check::new(
            "tokenizer files",
            CheckStatus::Pass,
            "all four tokenizer files are present".to_string(),
        ));
    } else {
        checks.push(Check::new(
            "tokenizer files",
            CheckStatus::Fail,
            format!(
                "missing {} in the tokenizer directory",
                inspection.tokenizer_missing.join(", ")
            ),
        ));
    }
}

/// The servable families this build recognizes, from the one list the
/// server config validates against.
fn servable_families() -> &'static [&'static str] {
    &crate::server::config::FAMILIES
}

fn identify_family(
    inspection: &ModelInspection,
    candidate: &Candidate,
    checks: &mut Vec<Check>,
) -> Option<String> {
    if let Some(family) = &candidate.family {
        if servable_families().contains(&family.as_str()) {
            checks.push(Check::new(
                "family",
                CheckStatus::Pass,
                format!("{family} (given) is recognized by this build"),
            ));
            return Some(family.clone());
        }
        checks.push(Check::new(
            "family",
            CheckStatus::Fail,
            format!(
                "family {family} is not recognized by this build (supported: {})",
                servable_families().join(", ")
            ),
        ));
        return None;
    }
    // Inference, deliberately narrow: only a `config.json` model_type that
    // a family's spec itself accepts. Anything else asks the user to name
    // the family rather than guessing at GB-scale consequences.
    let spec = crate::family::spec::QWEN3_6_35B_A3B;
    if inspection
        .config_model_types
        .iter()
        .any(|t| spec.model_types.contains(&t.as_str()))
    {
        checks.push(Check::new(
            "family",
            CheckStatus::Pass,
            format!(
                "config.json model_type matches {}",
                crate::family::qwen36::factory::FAMILY
            ),
        ));
        return Some(crate::family::qwen36::factory::FAMILY.to_string());
    }
    checks.push(Check::new(
        "family",
        CheckStatus::Fail,
        format!(
            "the family cannot be identified from this directory; pass --family (supported: {})",
            servable_families().join(", ")
        ),
    ));
    None
}

fn any_failed(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == CheckStatus::Fail)
}

fn any_warned(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == CheckStatus::Warn)
}

/// Inkling's header plan is available before a serving backend is wired up.
fn inkling2_mini_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    candidate: &Candidate,
    mut checks: Vec<Check>,
) -> PreflightReport {
    use crate::family::inkling2_mini;
    let plan = if candidate.kv_dtype.as_deref().is_some_and(|kv| kv != "bf16") {
        Err(crate::EngError::BadRequest(
            "Inkling 2 Mini currently prices BF16 KV".into(),
        ))
    } else {
        inkling2_mini::inspect(model_dir, candidate.context_limit)
    };
    let decisions = match plan {
        Ok(rows) => {
            checks.push(Check::new(
                "Inkling 2 Mini expert layout",
                CheckStatus::Pass,
                "fixed-loop config, shard inventory and mixed expert layout verified from headers"
                    .into(),
            ));
            rows
        }
        Err(error) => {
            checks.push(Check::new(
                "Inkling 2 Mini expert layout",
                CheckStatus::Fail,
                super::describe_error(&error),
            ));
            Vec::new()
        }
    };
    inkling2_mini_report(
        model_dir,
        tokenizer_dir,
        checks,
        decisions,
        "Inkling 2 Mini text execution is pending",
        "Use the header plan to size the text executor and qualify it against the reference model.",
    )
}

fn inkling2_mini_report(
    model_dir: &Path,
    tokenizer_dir: &Path,
    mut checks: Vec<Check>,
    decisions: Vec<PlanDecision>,
    execution_detail: &str,
    next_step: &str,
) -> PreflightReport {
    checks.push(Check::new(
        "text execution",
        CheckStatus::Fail,
        execution_detail.into(),
    ));
    PreflightReport {
        model_dir: model_dir.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        family: Some(crate::family::inkling2_mini::FAMILY.into()),
        verdict: Verdict::Unsupported,
        dimensions: Dimensions {
            compatibility: Compatibility::Unsupported,
            qualification: Qualification::Unverified,
            resource_fit: ResourceFit::Unknown,
        },
        checks,
        decisions,
        estimate: None,
        next_step: next_step.into(),
    }
}

fn shallow_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    _inspection: &ModelInspection,
    family: Option<String>,
    mut checks: Vec<Check>,
) -> PreflightReport {
    let (verdict, next_step) = if any_failed(&checks) || family.is_none() {
        (
            Verdict::Unsupported,
            "Fix the failed checks above; nothing here is loaded until they pass.".to_string(),
        )
    } else {
        checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            "no byte estimate is composed for this family yet; load-time admission will check the plan and refuse with byte counts if it does not fit".to_string(),
        ));
        (
            Verdict::Experimental,
            "The directory shape is plausible but this family is not qualified end to end from preflight; serve it and read the engine start lines for the admission result.".to_string(),
        )
    };
    let dimensions = Dimensions {
        compatibility: if verdict == Verdict::Unsupported {
            Compatibility::Unsupported
        } else {
            Compatibility::Unverified
        },
        qualification: Qualification::Unverified,
        resource_fit: ResourceFit::Unknown,
    };
    PreflightReport {
        model_dir: model_dir.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        family,
        verdict,
        dimensions,
        checks,
        decisions: Vec::new(),
        estimate: None,
        next_step,
    }
}

fn q36_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    inspection: &ModelInspection,
    candidate: &Candidate,
    facts: &HardwareFacts,
    backend: Option<crate::family::qwen36::settings::Backend>,
    mut checks: Vec<Check>,
) -> PreflightReport {
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};

    let family = Some(crate::family::qwen36::factory::FAMILY.to_string());
    let finish = |checks: Vec<Check>,
                  estimate: Option<Estimate>,
                  decisions: Vec<PlanDecision>,
                  qualification: Qualification| {
        finish_report(
            model_dir,
            tokenizer_dir,
            family.clone(),
            checks,
            estimate.as_ref(),
            None,
            decisions,
            qualification,
        )
    };

    let Some(backend) = backend else {
        checks.push(Check::new(
            "backend",
            CheckStatus::Fail,
            "no GPU backend is compiled into this build (build with --features native-cuda on Linux or native-mlx on Apple Silicon); this binary cannot serve this family".to_string(),
        ));
        return finish(checks, None, Vec::new(), Qualification::Unverified);
    };
    checks.push(Check::new(
        "backend",
        CheckStatus::Pass,
        format!(
            "{} backend is compiled in{}",
            match backend {
                Backend::Cuda => "CUDA",
                Backend::Mlx => "MLX",
            },
            match (backend, facts.cuda.device_count.as_ref()) {
                (Backend::Cuda, Some(n)) if *n == 0 => ", but no CUDA device is present",
                _ => "",
            }
        ),
    ));
    if backend == Backend::Cuda && facts.cuda.device_count.as_ref().is_some_and(|n| *n > 1) {
        checks.push(Check::new(
            "devices",
            CheckStatus::Info,
            "more than one CUDA device is present; this estimate is for device 0 alone (the default). A cuda_devices plan is admitted by its own placement planner at load, with per-device budgets this check does not reproduce".to_string(),
        ));
    }
    if backend == Backend::Cuda && facts.cuda.device_count.as_ref() == Some(&0) {
        checks.last_mut().expect("backend check").status = CheckStatus::Fail;
        return finish(checks, None, Vec::new(), Qualification::Unverified);
    }

    // The factory's own settings resolution: identical validation of the
    // candidate options, including every range and combination rule.
    let options = Q36Options {
        context_limit: candidate.context_limit,
        kv_dtype: candidate.kv_dtype.clone(),
        ..Q36Options::default()
    };
    let env = Q36Env::from_process(model_dir);
    let settings = match Q36Settings::new(model_dir, tokenizer_dir, Some(&options), &env, backend) {
        Ok(settings) => {
            checks.push(Check::new(
                "settings",
                CheckStatus::Pass,
                format!(
                    "context {}, kv_dtype {}, activation {} resolve under the {} rules",
                    settings.context_limit,
                    crate::family::qwen36::settings::kv_dtype_name(settings.kv),
                    crate::family::qwen36::settings::activation_name(settings.activation),
                    match backend {
                        Backend::Cuda => "CUDA",
                        Backend::Mlx => "MLX",
                    },
                ),
            ));
            settings
        }
        Err(error) => {
            checks.push(Check::new(
                "settings",
                CheckStatus::Fail,
                format!(
                    "the engine would refuse these settings: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, Vec::new(), Qualification::Unverified);
        }
    };

    // The tokenizer verdict is the loader's: bundle load under the
    // family's pin refuses any file that differs from the tested release.
    match crate::family::qwen36::settings::tokenizer_pin()
        .and_then(|pin| crate::tokenizer::TokenizerBundle::load(tokenizer_dir, &pin))
    {
        Ok(_) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Pass,
            "all four tokenizer files match the tested release (SHA-256 pin)".to_string(),
        )),
        Err(error) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Fail,
            format!(
                "the tokenizer would be refused at load: {}",
                super::describe_error(&error)
            ),
        )),
    }

    // Qualification of this exact combination, from the project's own
    // gates: the EXL3 pack's quality against the GGUF engine is not yet
    // measured (doc/explanation/status.md), so it stays experimental
    // until that gate passes.
    let qualification = match settings.weights {
        crate::family::qwen36::settings::WeightFormat::Exl3 => Qualification::Experimental,
        crate::family::qwen36::settings::WeightFormat::Gguf => Qualification::Qualified,
    };

    // Format vs. directory shape: the resolved weight format must have
    // its files here. Exact checkpoint identity stays the loader's check.
    use crate::family::qwen36::settings::WeightFormat;
    match settings.weights {
        WeightFormat::Gguf if backend == Backend::Cuda => {
            checks.push(pinned_gguf_check(model_dir, inspection));
        }
        // The MLX arm reads an MLX safetensors pack; `gguf` is only the unset default of
        // `weight_format`, which the MLX arm does not use.
        WeightFormat::Gguf
            if backend == Backend::Mlx
                && inspection.gguf_files.is_empty()
                && inspection.safetensors_files > 0 =>
        {
            checks.push(Check::new(
                "weight format",
                CheckStatus::Pass,
                format!(
                    "{} MLX safetensors shards are present; exact file identity is verified by the loader",
                    inspection.safetensors_files
                ),
            ));
        }
        WeightFormat::Gguf if inspection.gguf_files.is_empty() => checks.push(Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is gguf but no readable .gguf file is in the directory".to_string(),
        )),
        WeightFormat::Exl3 if inspection.safetensors_files == 0 => checks.push(Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is exl3 but no .safetensors shards are in the directory".to_string(),
        )),
        _ => checks.push(Check::new(
            "weight format",
            CheckStatus::Pass,
            format!(
                "{} checkpoint files are present; exact file identity is verified by the loader",
                crate::family::qwen36::settings::weight_format_name(settings.weights)
            ),
        )),
    }

    if any_failed(&checks) {
        return finish(checks, None, Vec::new(), qualification);
    }

    let spec = settings.spec;
    // The CUDA GGUF loader reads exactly the pinned file, so only its
    // bytes are the planning figure; other layouts use the whole
    // inventory of weight files found.
    let disk_bytes = if backend == Backend::Cuda && settings.weights == WeightFormat::Gguf {
        pinned_gguf_bytes(inspection).unwrap_or_else(|| inspection.weight_bytes())
    } else {
        inspection.weight_bytes()
    };
    // The EXL3 planning figure is the loader's own header walk, not the
    // bytes on disk: the walk counts the widening to F32 the load
    // performs, and refuses a pack of another size, rate, codebook or
    // layout exactly as the load would, before the first upload. The
    // walk needs the EXL3 weights layer, compiled with the CUDA backend
    // (and into every test build); a build without it never reaches an
    // EXL3 estimate: EXL3 loads need the CUDA backend, which such a
    // build either fails (no backend at all) or refuses in the settings
    // check above (the MLX arm has no EXL3).
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    let (weight_bytes, exl3_plan) = if settings.weights == WeightFormat::Exl3 {
        match crate::family::qwen36::exl3_plan(model_dir) {
            Ok(plan) => (plan.device_bytes, Some(plan)),
            Err(error) => {
                checks.push(Check::new(
                    "exl3 pack",
                    CheckStatus::Fail,
                    format!(
                        "the load would refuse this pack: {}",
                        super::describe_error(&error)
                    ),
                ));
                return finish(checks, None, Vec::new(), qualification);
            }
        }
    } else {
        (disk_bytes, None)
    };
    #[cfg(not(any(test, all(feature = "native-cuda", target_os = "linux"))))]
    let weight_bytes = disk_bytes;
    let estimate = match estimate_q36(&spec, &settings, weight_bytes, backend, facts) {
        Ok(estimate) => estimate,
        Err(error) => {
            checks.push(Check::new(
                "memory estimate",
                CheckStatus::Fail,
                format!(
                    "the estimate could not be computed: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, Vec::new(), qualification);
        }
    };
    let demand = estimate
        .weight_bytes
        .saturating_add(estimate.admission_charges_bytes);
    match (estimate.fit, estimate.borderline) {
        (ResourceFit::Fits, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Pass,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) fits the {} budget",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::DoesNotFit, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Fail,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) exceed the {} budget; load-time admission is authoritative",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, true) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated demand {} bytes lands inside the measurement band around the {} budget; load-time admission decides",
                demand, estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, false) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated {} bytes; no budget could be probed ({}), so fit is unverified",
                estimate.total_bytes, estimate.budget_kind
            ),
        )),
    }
    // Only the rows come from the family entry; the checks and the stored
    // estimate above stay this arm's. The entry re-runs the same settings
    // resolution, walk and search through the shared request — the second
    // call of one implementation, never a second copy of the math.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    let decisions = match exl3_plan.as_ref() {
        Some(_) => match crate::family::qwen36::preflight::preflight_rows(&RowsRequest {
            model_dir,
            tokenizer_dir,
            context: candidate.context_limit,
            kv_dtype: candidate.kv_dtype.as_deref(),
            prefill_rows: candidate.prefill_rows,
            facts,
        }) {
            Ok((rows, _note, _fit)) => rows,
            Err(error) => {
                checks.push(Check::new(
                    "exl3 pack",
                    CheckStatus::Fail,
                    format!(
                        "the load would refuse this pack: {}",
                        super::describe_error(&error)
                    ),
                ));
                Vec::new()
            }
        },
        None => Vec::new(),
    };
    #[cfg(not(any(test, all(feature = "native-cuda", target_os = "linux"))))]
    let decisions = Vec::new();
    finish(checks, Some(estimate), decisions, qualification)
}

/// The Flint deep path: the only family with a three-way load decision
/// (all experts resident, an automatically sized bank, or a refusal), so
/// its preflight composes the placement the loader would choose from the
/// same pure functions (`family::flint::budget` and the header-level
/// resident audit) and reports it as decisions. `cuda_backend` is this
/// build's own; tests inject.
fn flint_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    _inspection: &ModelInspection,
    candidate: &Candidate,
    facts: &HardwareFacts,
    cuda_backend: bool,
    mut checks: Vec<Check>,
) -> PreflightReport {
    use crate::family::flint::settings::FlintOptions;

    let family = Some(crate::family::flint::FAMILY.to_string());
    let finish = |checks: Vec<Check>,
                  estimate: Option<Estimate>,
                  decisions: Vec<PlanDecision>,
                  qualification: Qualification| {
        finish_report(
            model_dir,
            tokenizer_dir,
            family.clone(),
            checks,
            estimate.as_ref(),
            None,
            decisions,
            qualification,
        )
    };

    if !cuda_backend {
        checks.push(Check::new(
            "backend",
            CheckStatus::Fail,
            "no CUDA backend is compiled into this build (build with --features native-cuda on Linux); this binary cannot serve Flint".to_string(),
        ));
        return finish(checks, None, Vec::new(), Qualification::Unverified);
    }
    checks.push(Check::new(
        "backend",
        CheckStatus::Pass,
        match facts.cuda.device_count.as_ref() {
            Some(count) if *count > 0 => {
                format!("CUDA backend is compiled in; {count} device(s) present")
            }
            Some(_) => "CUDA backend is compiled in, but no CUDA device is present".to_string(),
            None => "CUDA backend is compiled in".to_string(),
        },
    ));
    if facts.cuda.device_count.as_ref() == Some(&0) {
        checks.last_mut().expect("backend check").status = CheckStatus::Fail;
        return finish(checks, None, Vec::new(), Qualification::Unverified);
    }

    // The candidate settings resolve exactly as a config file would: the
    // CUDA default plus the overrides, validated by the factory's own
    // plan resolution (context cap, kv profile, chunk range).
    let mut options = FlintOptions::cuda_default();
    if let Some(context) = candidate.context_limit {
        options.context_limit = context;
    }
    if let Some(profile) = &candidate.kv_dtype {
        options.kv_profile = profile.clone();
    }
    match options.plan() {
        Ok(_) => checks.push(Check::new(
            "settings",
            CheckStatus::Pass,
            format!(
                "context {}, kv_profile {}, prefill_chunk_size {} resolve under the Flint CUDA rules",
                options.context_limit, options.kv_profile, options.prefill_chunk_size
            ),
        )),
        Err(error) => {
            checks.push(Check::new(
                "settings",
                CheckStatus::Fail,
                format!(
                    "the engine would refuse these settings: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, Vec::new(), Qualification::Unverified);
        }
    }

    // The tokenizer verdict is the loader's: bundle load under the
    // family's pin refuses any file that differs from the tested release.
    match crate::tokenizer::TokenizerBundle::load(
        tokenizer_dir,
        &crate::family::flint::factory::tokenizer_pin(),
    ) {
        Ok(_) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Pass,
            "all four tokenizer files match the tested release (SHA-256 pin)".to_string(),
        )),
        Err(error) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Fail,
            format!(
                "the tokenizer would be refused at load: {}",
                super::describe_error(&error)
            ),
        )),
    }

    // Qualification: the served receipts in doc/explanation/status.md
    // cover this combination (default boot, 131072/f16, banked).
    let qualification = Qualification::Qualified;

    if any_failed(&checks) {
        return finish(checks, None, Vec::new(), qualification);
    }

    // The environment switches, read once here exactly as the loader
    // reads them at load.
    #[cfg_attr(
        not(any(test, all(feature = "native-cuda", target_os = "linux"))),
        allow(unused_variables)
    )]
    let env = crate::family::flint::budget::FlintEnv::capture();
    // The resident audit needs a build with the CUDA weights layer (or a
    // test build); a default-feature build never reaches here because its
    // backend check failed above, but the pair stays typed either way.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    let pair = crate::family::flint::exl3_resident::audit_plan(model_dir, u64::MAX)
        .and_then(|audit| flint_estimate(&audit, &options, &env, facts));
    #[cfg(not(any(test, all(feature = "native-cuda", target_os = "linux"))))]
    let pair: Result<(Estimate, Vec<PlanDecision>), crate::EngError> = Err(
        crate::EngError::BadRequest(
            "this build cannot compose the Flint memory estimate; build with --features native-cuda on Linux".into(),
        ),
    );
    let (estimate, decisions) = match pair {
        Ok(pair) => pair,
        Err(error) => {
            checks.push(Check::new(
                "memory estimate",
                CheckStatus::Fail,
                format!(
                    "the estimate could not be computed: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, Vec::new(), qualification);
        }
    };
    let demand = estimate
        .weight_bytes
        .saturating_add(estimate.admission_charges_bytes);
    match (estimate.fit, estimate.borderline) {
        (ResourceFit::Fits, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Pass,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) fits the {} budget",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::DoesNotFit, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Fail,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) exceed the {} budget; load-time admission is authoritative",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, true) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated demand {} bytes lands inside the measurement band around the {} budget; load-time admission decides",
                demand, estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, false) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated {} bytes; no budget could be probed ({}), so fit is unverified",
                estimate.total_bytes, estimate.budget_kind
            ),
        )),
    }
    finish(checks, Some(estimate), decisions, qualification)
}

/// The Flint estimate and its decisions, from the same pure functions the
/// loader runs: the header-level resident audit for the weight terms, the
/// KV plan and the closed-form work reserve for the charges, and the
/// budget switch parsers for the request. Only a CUDA-capable build (or
/// a test build) has the resident audit; the fallback below keeps the
/// default-feature build honest.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn flint_estimate(
    audit: &crate::family::flint::exl3_resident::AuditPlan,
    options: &crate::family::flint::settings::FlintOptions,
    env: &crate::family::flint::budget::FlintEnv,
    facts: &HardwareFacts,
) -> Result<(Estimate, Vec<PlanDecision>), crate::EngError> {
    use crate::family::flint::budget::{
        self, BankPlan, BankReason, BankRequest, CpuMoeConfig, FLINT_HEADROOM_BYTES,
        MODEL_CONTEXT_LIMIT,
    };

    let plan = options.plan()?;
    let (k_row, v_row) = plan.per_plane_row_bytes()?;
    let row_bytes = k_row + v_row;
    let kv_bytes = u64::try_from(plan.bytes(row_bytes)?)
        .map_err(|_| crate::EngError::MemoryPlan("Flint KV estimate overflows u64".to_string()))?;
    let work = budget::work_reserve_closed_form(
        options.prefill_chunk_size,
        budget::alloc_cache_reserve(env.alloc_cache_enabled()),
    )?;
    let headroom = FLINT_HEADROOM_BYTES;

    // The request the loader composes: the config's CPU-expert split
    // (from the options) and the operator's environment switches.
    let cpu = CpuMoeConfig::new(
        options.cpu_moe_offload_layers,
        options.cpu_moe_split_experts,
        options.cpu_moe_threads,
    )?;
    let (mode, request) = budget::cpu_moe_bank(
        cpu.enabled().then_some(cpu),
        env.expert_cache_mode()?,
        env.bank_request()?,
    )?;
    // The offload rule the loader charges: CPU-expert layers keep no
    // device slots, and an active CPU expert path lowers a full bank by
    // one (see `budget::banked_slots`).
    let cpu_active = cpu.enabled();
    let offloaded = if cpu_active { cpu.offload_layers() } else { 0 };
    let estimate_overflow =
        || crate::EngError::MemoryPlan("Flint estimate byte count overflows u64".into());

    let free = match &facts.cuda.first_device_free_bytes {
        Probe::Measured(free) => Some(*free),
        Probe::Unavailable(_) => None,
    };
    let kv_at = |context: u32| -> Result<u64, crate::EngError> {
        let mut probe = options.clone();
        probe.context_limit = context;
        u64::try_from(probe.plan()?.bytes(row_bytes)?)
            .map_err(|_| crate::EngError::MemoryPlan("Flint KV estimate overflows u64".into()))
    };
    let resident_admits_at = |context: u32, free: u64| -> bool {
        let kv = kv_at(context).ok();
        kv.is_some_and(|kv| budget::admits(audit.resident_bytes, kv, work, headroom, free).is_ok())
    };
    let resident_fits = free
        .map(|free| resident_admits_at(options.context_limit, free))
        .unwrap_or(false);
    let placement = budget::plan_bank(mode, request, resident_fits)?;

    // Whether the CPU-expert rule lowered a full bank (surfaces in the
    // placement decision text, as the loader's own load note does).
    let mut lowered = false;
    let (weight_bytes, slots, margin, bank_reason) = match placement {
        BankPlan::Resident => (audit.resident_bytes, None, 0, None),
        BankPlan::Bank(reason) => {
            let free = free.ok_or_else(|| {
                crate::EngError::BadRequest(
                    "the bank plan needs the measured free bytes of device 0".into(),
                )
            })?;
            let per_slot = audit.per_slot(offloaded)?;
            let bank =
                budget::bank_budget(free, audit.arena_base, per_slot, kv_bytes, work, headroom)?;
            let floor =
                budget::cache_prefill_floor(crate::family::flint::settings::PREFILL_MAX_ROWS)?;
            let (slots, cap) = budget::banked_slots(request, bank, floor, cpu_active)?;
            lowered = cap;
            let weight_bytes = audit
                .arena_base
                .checked_add(
                    u64::try_from(slots)
                        .map_err(|_| estimate_overflow())?
                        .checked_mul(per_slot)
                        .ok_or_else(estimate_overflow)?,
                )
                .ok_or_else(estimate_overflow)?;
            (
                weight_bytes,
                Some(slots),
                budget::BANK_MARGIN_BYTES,
                Some(reason),
            )
        }
    };

    let reserve = crate::family::device_budget::DeviceReserve::new(
        crate::family::admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES,
    )
    .push("kv cache", kv_bytes)
    .push("work reserve", work)
    .push("load headroom", headroom)
    .push("bank margin", margin);
    // The receipt is the reserve's own charges (four terms when a bank
    // margin applies, three when not) with the load-time scratch line
    // appended last, exactly as before.
    let charge_terms = || {
        let mut terms: Vec<(String, u64)> = reserve
            .charges()
            .iter()
            .map(|&(label, bytes)| (label.to_string(), bytes))
            .collect();
        if margin == 0 {
            terms.retain(|(name, _)| name != "bank margin");
        }
        terms.push((
            "device scratch (excluded here; charged at load)".to_string(),
            0,
        ));
        terms
    };
    let charges = reserve.checked_total().ok_or_else(estimate_overflow)?;
    let total = weight_bytes
        .checked_add(charges)
        .ok_or_else(estimate_overflow)?;

    let band = i128::from(reserve.band_bytes());
    let (fit, borderline, largest_fitting) = match free {
        None => (ResourceFit::Unknown, false, None),
        Some(free) => {
            let slack = i128::from(free) - i128::from(weight_bytes) - i128::from(charges);
            let (fit, borderline) = band_verdict(slack, band);
            // The receipt sentence for an automatically banked load: the
            // largest context the all-resident plan would have admitted.
            let largest_fitting = if matches!(placement, BankPlan::Bank(BankReason::Auto)) {
                largest_resident_context(options, &resident_admits_at, free)
            } else {
                None
            };
            (fit, borderline, largest_fitting)
        }
    };

    let mib = super::mib;
    let expert_placement = match (slots, bank_reason) {
        // A banked placement always carries its reason; the pair cannot
        // be split by construction.
        (Some(_), None) => unreachable!("bank placement without a reason"),
        (None, _) => PlanDecision {
            name: "expert_placement".to_string(),
            chosen: format!(
                "resident: all 256 experts per layer on the GPU ({} weights)",
                mib(audit.resident_bytes)
            ),
            alternatives: vec![PlanAlternative {
                option: "bank".to_string(),
                verdict: "not needed".to_string(),
                reason: "the all-resident plan fits the measured free bytes".to_string(),
            }],
            fixed_at: PlanFixity::Load,
        },
        (Some(slots), Some(reason)) => PlanDecision {
            name: "expert_placement".to_string(),
            chosen: format!(
                "bank: {slots} of 256 experts per layer on the GPU, all 256 in pinned host memory ({}{})",
                budget::bank_note(options.context_limit, slots, reason.as_str()),
                if lowered {
                    "; lowered from 256 so the pinned host store exists"
                } else {
                    ""
                }
            ),
            alternatives: largest_fitting
                .map(|context| PlanAlternative {
                    option: "resident".to_string(),
                    verdict: "does not fit at this context".to_string(),
                    reason: format!(
                        "all-resident would admit at most {context} tokens at this profile"
                    ),
                })
                .into_iter()
                .chain(std::iter::once(PlanAlternative {
                    option: "resident".to_string(),
                    verdict: if resident_fits {
                        "fits"
                    } else {
                        "does not fit"
                    }
                    .to_string(),
                    reason: format!(
                        "all-resident weights {} + charges {} vs free {}",
                        mib(audit.resident_bytes),
                        mib(kv_bytes + work + headroom),
                        mib(free.unwrap_or(0)),
                    ),
                }))
                .collect(),
            fixed_at: PlanFixity::Load,
        },
    };
    let decisions = vec![
        expert_placement,
        PlanDecision {
            name: "expert_cache".to_string(),
            chosen: format!(
                "{} ({CACHE_OR_DEFAULT})",
                match mode {
                    crate::family::flint::budget::ExpertCacheMode::Auto => "auto",
                    crate::family::flint::budget::ExpertCacheMode::On => "on",
                    crate::family::flint::budget::ExpertCacheMode::Off => "off",
                },
                CACHE_OR_DEFAULT = if env.expert_cache.is_some() {
                    crate::family::flint::budget::CACHE_VAR
                } else {
                    "unset"
                }
            ),
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "expert_slots".to_string(),
            chosen: match request {
                BankRequest::Fit => "fit: as many as the device budget leaves".to_string(),
                BankRequest::Exactly(count) => format!("exactly {count}"),
            },
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "context_limit".to_string(),
            chosen: options.context_limit.to_string(),
            alternatives: vec![PlanAlternative {
                option: MODEL_CONTEXT_LIMIT.to_string(),
                verdict: "cap".to_string(),
                reason: "the model's own context limit; no profile exceeds it".to_string(),
            }],
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "kv_profile".to_string(),
            chosen: format!("{} ({} B/position)", options.kv_profile, row_bytes),
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "prefill_chunk_size".to_string(),
            chosen: format!(
                "{} (bank floor {} experts per layer)",
                options.prefill_chunk_size,
                budget::FLOOR_EXPERTS_PER_ROW * options.prefill_chunk_size as usize
            ),
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "cpu_moe".to_string(),
            chosen: if cpu.enabled() {
                format!(
                    "on: {} offloaded layers, {} split experts, {} threads",
                    cpu.offload_layers(),
                    cpu.split_experts(),
                    cpu.threads()
                )
            } else {
                "off".to_string()
            },
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
        PlanDecision {
            name: "layer_major".to_string(),
            chosen: {
                let order = env.prefill_order(options, &cpu)?;
                let name = match order.mode {
                    crate::family::flint::budget::LayerMajor::Off => "off (token-major)",
                    crate::family::flint::budget::LayerMajor::Tiles => "tiles",
                    crate::family::flint::budget::LayerMajor::Jobs => "jobs",
                };
                format!(
                    "{name} ({}: {}){}{}",
                    if order.defaulted {
                        "default"
                    } else {
                        "explicit"
                    },
                    order.why,
                    if order.hilo { ", hi/lo split on" } else { "" },
                    if order.mode == crate::family::flint::budget::LayerMajor::Jobs {
                        "; the fit estimate does not include the jobs pass's extra work reserve (about 0.6 GB); an open that does not fit retries on the token-major walk"
                    } else {
                        ""
                    }
                )
            },
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Start,
        },
    ];

    let estimate = Estimate {
        weight_bytes,
        kv_bytes_per_position: row_bytes as u64,
        context_limit: options.context_limit,
        kv_bytes,
        fixed_state_bytes: 0,
        total_bytes: total,
        budget_kind: match &facts.cuda.first_device_free_bytes {
            Probe::Measured(_) => "free VRAM of device 0, measured now, minus the runtime admission charges below (estimate; load-time admission is authoritative)".to_string(),
            Probe::Unavailable(reason) => format!(
                "free VRAM of device 0; unavailable: {reason}"
            ),
        },
        budget_free_bytes: free,
        budget_limit_bytes: free,
        budget_limit_is_policy_share: false,
        largest_fitting_context: largest_fitting,
        admission_charges_bytes: charges,
        admission_charge_terms: charge_terms(),
        fit,
        borderline,
    };
    Ok((estimate, decisions))
}

/// The report layer's mapping of the one shared band rule (see
/// `family::admission_sketch`) onto the serialized fit: borderline becomes
/// `Unknown` plus the flag the report renders.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn band_verdict(slack: i128, band: i128) -> (ResourceFit, bool) {
    match crate::family::admission_sketch::band_verdict(slack, band) {
        crate::family::admission_sketch::BandVerdict::Fits => (ResourceFit::Fits, false),
        crate::family::admission_sketch::BandVerdict::Borderline => (ResourceFit::Unknown, true),
        crate::family::admission_sketch::BandVerdict::DoesNotFit => {
            (ResourceFit::DoesNotFit, false)
        }
    }
}

/// The largest context (a multiple of 1024, capped at the requested
/// context) whose all-resident plan fits `free` under the same admission
/// mirror the estimate uses. `None` when not even a minimal context fits.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn largest_resident_context(
    options: &crate::family::flint::settings::FlintOptions,
    resident_admits_at: &dyn Fn(u32, u64) -> bool,
    free: u64,
) -> Option<u32> {
    crate::family::device_budget::largest_fitting_linear(options.context_limit, 1024, &|context| {
        resident_admits_at(context, free)
    })
}

/// The CUDA checkpoint check, through the loader's own gates: the
/// `Q36_ALLOW_UNPINNED_GGUF` switch decides the selection, then
/// `check_gguf_identity` pins the curated file's exact size (the
/// full-file SHA-256 stays the loader's opt-in, since hashing 20 GB
/// is not a preflight default; an unpinned checkpoint has no pinned
/// size or digest and is validated by the model open). Only a CUDA
/// build has the gate at all — and a CUDA backend implies that
/// build — so the non-CUDA arm below is reachable only from tests
/// that inject the backend.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn pinned_gguf_check(model_dir: &Path, _inspection: &ModelInspection) -> Check {
    use crate::family::qwen36::cuda::start::ALLOW_UNPINNED_VAR;
    gguf_admission_check(model_dir, std::env::var(ALLOW_UNPINNED_VAR).ok().as_deref())
}

/// The pure core of [`pinned_gguf_check`]: the raw switch value arrives resolved by the
/// caller, so the branches (a bad spelling, a refused selection, the pinned and the admitted
/// checkpoint) test without touching process-global state.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn gguf_admission_check(model_dir: &Path, raw: Option<&str>) -> Check {
    use crate::family::qwen36::cuda::start::{
        ALLOW_UNPINNED_VAR, GGUF_NAME, allow_unpinned_wanted, check_gguf_identity, select_gguf,
    };
    let allow_unpinned = match allow_unpinned_wanted(raw) {
        Ok(wanted) => wanted,
        Err(error) => {
            return Check::new(
                "weight format",
                CheckStatus::Fail,
                format!(
                    "the loader would refuse this start: {}",
                    super::describe_error(&error)
                ),
            );
        }
    };
    let (gguf, pinned) = match select_gguf(model_dir, allow_unpinned) {
        Ok(pair) => pair,
        Err(error) => {
            return Check::new(
                "weight format",
                CheckStatus::Fail,
                format!(
                    "the loader would refuse this checkpoint: {}",
                    super::describe_error(&error)
                ),
            );
        }
    };
    match check_gguf_identity(&gguf, false, pinned) {
        Ok(()) => Check::new(
            "weight format",
            CheckStatus::Pass,
            if pinned {
                format!("the pinned CUDA checkpoint {GGUF_NAME} is present at its pinned size")
            } else {
                format!(
                    "{} is not the pinned file; it is served unless {ALLOW_UNPINNED_VAR}=0, and the \
                     model open validates it",
                    gguf.display()
                )
            },
        ),
        Err(error) => Check::new(
            "weight format",
            CheckStatus::Fail,
            format!(
                "the loader would refuse this checkpoint: {}",
                super::describe_error(&error)
            ),
        ),
    }
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn pinned_gguf_check(_model_dir: &Path, inspection: &ModelInspection) -> Check {
    if inspection.gguf_files.is_empty() {
        Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is gguf but no readable .gguf file is in the directory".to_string(),
        )
    } else {
        Check::new(
            "weight format",
            CheckStatus::Warn,
            "the pinned-file identity check lives in the CUDA build; this binary cannot run it"
                .to_string(),
        )
    }
}

/// The pinned file's bytes from the inventory, when this build knows
/// the pinned name (same gate as `pinned_gguf_check`).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn pinned_gguf_bytes(inspection: &ModelInspection) -> Option<u64> {
    use crate::family::qwen36::cuda::start::GGUF_NAME;
    inspection
        .gguf_files
        .iter()
        .find(|fact| fact.file == GGUF_NAME)
        .map(|fact| fact.file_bytes)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn pinned_gguf_bytes(_inspection: &ModelInspection) -> Option<u64> {
    None
}

fn q36_dimensions(
    checks: &[Check],
    estimate: Option<&Estimate>,
    qualification: Qualification,
) -> Dimensions {
    // A failed memory estimate is a resource fact, not a compatibility
    // one; every other failed check here means this build cannot serve
    // the directory as asked.
    let incompatible = checks
        .iter()
        .any(|c| c.status == CheckStatus::Fail && c.name != "memory estimate");
    let resource_fit = match estimate {
        Some(estimate) => estimate.fit,
        None => ResourceFit::Unknown,
    };
    Dimensions {
        compatibility: if incompatible {
            Compatibility::Unsupported
        } else {
            Compatibility::Supported
        },
        qualification,
        resource_fit,
    }
}

fn q36_verdict(
    checks: &[Check],
    estimate: Option<&Estimate>,
    qualification: Qualification,
) -> (Verdict, String) {
    if let Some(estimate) = estimate
        && estimate.fit == ResourceFit::DoesNotFit
    {
        let next = match estimate.largest_fitting_context {
            Some(fitting) if fitting >= 2 => format!(
                "Free memory, or serve with a smaller context: the largest estimated fitting context is {fitting} (requested {}). Nothing was changed for you.",
                estimate.context_limit
            ),
            _ => "Free memory or use a smaller checkpoint: even a minimal context does not fit the probed budget.".to_string(),
        };
        return (Verdict::ResourceConstrained, next);
    }
    if any_failed(checks) {
        return (
            Verdict::Unsupported,
            "Fix the failed checks above; preflight ran the same validators the loader runs, so it would refuse the same way.".to_string(),
        );
    }
    if let Some(estimate) = estimate
        && estimate.borderline
    {
        return (
            Verdict::ReadyWithAdjustments,
            "The estimate lands inside the measurement band around the budget, so load-time admission decides: try serving, or pick a smaller context for certainty.".to_string(),
        );
    }
    if qualification == Qualification::Experimental {
        return (
            Verdict::Experimental,
            "Servable, but this combination has not passed its quality gate; treat output quality as unmeasured against the qualified path.".to_string(),
        );
    }
    if any_warned(checks) {
        return (
            Verdict::ReadyWithAdjustments,
            "Read the warnings above, then serve; load-time admission re-checks the plan."
                .to_string(),
        );
    }
    (
        Verdict::Ready,
        "Serve it: enginesrv --config <file> with this directory as model_dir (load-time admission re-checks the plan).".to_string(),
    )
}

fn estimate_q36(
    spec: &crate::family::spec::FamilySpec,
    settings: &crate::family::qwen36::settings::Q36Settings,
    weight_bytes: u64,
    backend: crate::family::qwen36::settings::Backend,
    facts: &HardwareFacts,
) -> Result<Estimate, crate::EngError> {
    use crate::family::admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES;
    use crate::family::qwen36::admission_sketch as sketch;
    use crate::family::qwen36::settings::Backend;

    // The one CUDA demand model: the same builder the family's preflight
    // rows call, so the estimate and the rows cannot drift apart.
    let (per_position, state_bytes, cuda_demand_at) =
        sketch::cuda_demand_model(spec, settings.kv, settings.activation)?;
    let kv_bytes = per_position
        .checked_mul(u64::from(settings.context_limit))
        .ok_or_else(|| {
            crate::EngError::MemoryPlan("preflight KV estimate overflows u64".to_string())
        })?;
    let weights = weight_bytes;
    let total = weights
        .checked_add(kv_bytes)
        .and_then(|sum| sum.checked_add(state_bytes))
        .ok_or_else(|| {
            crate::EngError::MemoryPlan("preflight total estimate overflows u64".to_string())
        })?;

    let (budget_kind, free, limit_is_policy_share) = match backend {
        Backend::Cuda => (
            "free VRAM of device 0, measured now, minus the runtime admission charges below (estimate; load-time admission is authoritative)".to_string(),
            facts.cuda.first_device_free_bytes.clone(),
            false,
        ),
        Backend::Mlx => (
            "available unified memory, measured now".to_string(),
            facts.host.mem_available_bytes.clone(),
            true,
        ),
    };
    let policy = MemoryPolicy::default_share();

    let (free_bytes, limit_bytes, charges, charge_terms, fit, borderline, cuda_proposal) =
        match (&free, backend) {
            (Probe::Measured(free), Backend::Cuda) => {
                let demand = cuda_demand_at(settings.context_limit)?;
                // The charge total is the whole demand (state included); the
                // receipt lists only the five admission terms, as it always has.
                let charges = demand.total_after_weights()?;
                let reserve = crate::family::device_budget::DeviceReserve::new(
                    ADMISSION_UNCERTAINTY_BAND_BYTES,
                )
                .push("staged kv8 rows", demand.staging_bytes)
                .push("part charge", demand.part_charge_bytes)
                .push("spare minimum", demand.spare_min_bytes)
                .push("admission margin", demand.margin_bytes)
                .push("runtime overhead floor", demand.overhead_floor_bytes);
                // The verdict and the proposal are the one shared call the
                // preflight rows also make; the receipt above is the estimate's
                // own composition of the same demand's terms.
                let (verdict, largest_fitting) = sketch::cuda_verdict_and_proposal(
                    *free,
                    weights,
                    &cuda_demand_at,
                    per_position,
                    settings.context_limit,
                )?;
                let (fit, borderline) = match verdict {
                    crate::family::admission_sketch::BandVerdict::Fits => {
                        (ResourceFit::Fits, false)
                    }
                    crate::family::admission_sketch::BandVerdict::Borderline => {
                        (ResourceFit::Unknown, true)
                    }
                    crate::family::admission_sketch::BandVerdict::DoesNotFit => {
                        (ResourceFit::DoesNotFit, false)
                    }
                };
                (
                    Some(*free),
                    Some(*free),
                    charges,
                    reserve
                        .charges()
                        .iter()
                        .map(|&(label, bytes)| (label.to_string(), bytes))
                        .collect(),
                    fit,
                    borderline,
                    largest_fitting,
                )
            }
            (Probe::Measured(free), Backend::Mlx) => {
                let limit = policy.limit(*free);
                let fit = if total <= limit {
                    ResourceFit::Fits
                } else {
                    ResourceFit::DoesNotFit
                };
                (Some(*free), Some(limit), 0, Vec::new(), fit, false, None)
            }
            (Probe::Unavailable(_), _) => {
                (None, None, 0, Vec::new(), ResourceFit::Unknown, false, None)
            }
        };

    // A proposal only when the requested context is not a confident fit; the
    // search charges the same paper demand the runtime ledger charges, band
    // included. Not even a minimal context fitting yields None, and the
    // verdict then says so in its own words.
    let proposal_needed = per_position > 0 && (fit == ResourceFit::DoesNotFit || borderline);
    let largest_fitting = if backend == Backend::Cuda {
        cuda_proposal
    } else if !proposal_needed {
        None
    } else {
        let for_kv = limit_bytes
            .unwrap_or(0)
            .saturating_sub(weights.saturating_add(state_bytes));
        let positions = for_kv / per_position;
        (positions >= 2).then_some(u32::try_from(positions).unwrap_or(u32::MAX))
    };
    Ok(Estimate {
        weight_bytes: weights,
        kv_bytes_per_position: per_position,
        context_limit: settings.context_limit,
        kv_bytes,
        fixed_state_bytes: state_bytes,
        total_bytes: total,
        budget_kind: match &free {
            Probe::Measured(_) => budget_kind,
            Probe::Unavailable(reason) => format!("{budget_kind}; unavailable: {reason}"),
        },
        budget_free_bytes: free_bytes,
        budget_limit_bytes: limit_bytes,
        budget_limit_is_policy_share: limit_is_policy_share,
        largest_fitting_context: largest_fitting,
        admission_charges_bytes: charges,
        admission_charge_terms: charge_terms,
        fit,
        borderline,
    })
}

#[cfg(test)]
mod tests {
    use super::largest_resident_context;

    /// c0 goldens (P4): the Flint resident-context scan. It starts at 1024,
    /// steps by 1024, and STOPS at the first non-admitting step - a later gap
    /// never rescues a larger value - and never returns the limit itself.
    #[test]
    fn largest_resident_context_golden_scan() {
        let opts = |limit: u32| {
            let mut options = crate::family::flint::settings::FlintOptions::cuda_default();
            options.context_limit = limit;
            options
        };
        // Monotone cut at 50_000 (off-lattice): last admitting step 49_152.
        let cut = |context: u32, _free: u64| context <= 50_000;
        assert_eq!(
            largest_resident_context(&opts(131_072), &cut, 1 << 40),
            Some(49_152)
        );
        // A gap (20_480..=30_720 refuses, above admits): the scan STOPS in the
        // gap and returns 19_456, not the later larger value.
        let gap = |context: u32, _free: u64| !(20_480..=30_720).contains(&context);
        assert_eq!(
            largest_resident_context(&opts(131_072), &gap, 1 << 40),
            Some(19_456)
        );
        // First step refuses, or the limit is below the first step.
        assert_eq!(
            largest_resident_context(&opts(131_072), &|_, _| false, 1 << 40),
            None
        );
        assert_eq!(
            largest_resident_context(&opts(1023), &|_, _| true, 1 << 40),
            None
        );
        // A cut exactly on the lattice returns that step.
        let on_lattice = |context: u32, _free: u64| context <= 30_720;
        assert_eq!(
            largest_resident_context(&opts(131_072), &on_lattice, 1 << 40),
            Some(30_720)
        );
    }

    use std::path::Path;

    use super::{
        Candidate, Check, CheckStatus, Compatibility, Estimate, PlanAlternative, PlanDecision,
        PlanFixity, Qualification, ResourceFit, Verdict, q36_dimensions, q36_verdict,
    };

    fn estimate(total: u64, limit: Option<u64>, fitting: Option<u32>) -> Estimate {
        let fit = match limit {
            Some(limit) if total <= limit => ResourceFit::Fits,
            Some(_) => ResourceFit::DoesNotFit,
            None => ResourceFit::Unknown,
        };
        Estimate {
            weight_bytes: 100,
            kv_bytes_per_position: 10,
            context_limit: 1000,
            kv_bytes: total.saturating_sub(100),
            fixed_state_bytes: 0,
            total_bytes: total,
            budget_kind: "test budget".into(),
            budget_free_bytes: limit,
            budget_limit_bytes: limit,
            budget_limit_is_policy_share: true,
            largest_fitting_context: fitting,
            admission_charges_bytes: 0,
            admission_charge_terms: Vec::new(),
            fit,
            borderline: false,
        }
    }

    #[test]
    fn an_over_budget_estimate_is_resource_constrained_with_a_proposal() {
        let (verdict, next) = q36_verdict(
            &[],
            Some(&estimate(2000, Some(1000), Some(90))),
            Qualification::Qualified,
        );
        assert_eq!(verdict, Verdict::ResourceConstrained);
        assert!(
            next.contains("largest estimated fitting context is 90"),
            "{next}"
        );
        assert!(next.contains("Nothing was changed"), "{next}");
    }

    #[test]
    fn failures_outrank_warnings_and_warnings_downgrade_ready() {
        let check = |status| Check::new("x", status, String::new());
        assert_eq!(
            q36_verdict(&[check(CheckStatus::Pass)], None, Qualification::Qualified).0,
            Verdict::Ready
        );
        assert_eq!(
            q36_verdict(
                &[check(CheckStatus::Pass), check(CheckStatus::Warn)],
                None,
                Qualification::Qualified
            )
            .0,
            Verdict::ReadyWithAdjustments
        );
        assert_eq!(
            q36_verdict(
                &[check(CheckStatus::Pass), check(CheckStatus::Fail)],
                None,
                Qualification::Qualified
            )
            .0,
            Verdict::Unsupported
        );
    }

    // ----- the rows arm -----

    fn rows_stub_ok(
        _request: &super::RowsRequest,
    ) -> Result<(Vec<PlanDecision>, String, super::RowsFit), crate::EngError> {
        Ok((
            vec![PlanDecision {
                name: "context".to_string(),
                chosen: "4096 positions".to_string(),
                alternatives: Vec::new(),
                fixed_at: PlanFixity::Start,
            }],
            "a stub note".to_string(),
            super::RowsFit::Fits,
        ))
    }

    fn rows_stub_refused(
        _request: &super::RowsRequest,
    ) -> Result<(Vec<PlanDecision>, String, super::RowsFit), crate::EngError> {
        Err(crate::EngError::BadRequest("the stub refuses".to_string()))
    }

    fn rows_stub_constrained(
        _request: &super::RowsRequest,
    ) -> Result<(Vec<PlanDecision>, String, super::RowsFit), crate::EngError> {
        Ok((
            vec![
                PlanDecision {
                    name: "context".to_string(),
                    chosen: "262144 positions".to_string(),
                    alternatives: vec![PlanAlternative {
                        option: "97280 positions".to_string(),
                        verdict: "fits".to_string(),
                        reason: "largest multiple of 1024 the same plan admits".to_string(),
                    }],
                    fixed_at: PlanFixity::Start,
                },
                PlanDecision {
                    name: "K/V".to_string(),
                    chosen: "F16 0.05 GiB".to_string(),
                    alternatives: Vec::new(),
                    fixed_at: PlanFixity::Start,
                },
            ],
            "a stub note".to_string(),
            super::RowsFit::Constrained {
                requested: 262_144,
                largest: 97_280,
            },
        ))
    }

    fn rows_stub_no_facts(
        _request: &super::RowsRequest,
    ) -> Result<(Vec<PlanDecision>, String, super::RowsFit), crate::EngError> {
        Ok((
            Vec::new(),
            "the device's free memory could not be probed: no driver".to_string(),
            super::RowsFit::Unknown,
        ))
    }

    #[test]
    fn the_rows_arm_renders_the_source_rows_as_the_decisions() {
        let report = super::rows_arm(
            Path::new("/m"),
            Path::new("/t"),
            Some("a family".to_string()),
            rows_stub_ok,
            &Candidate::default(),
            &cuda_facts(24 << 30),
            Qualification::Experimental,
            Vec::new(),
        );
        assert_eq!(report.family.as_deref(), Some("a family"));
        assert_eq!(report.decisions.len(), 1);
        assert_eq!(report.decisions[0].chosen, "4096 positions");
        assert_eq!(check_named(&report, "load plan").status, CheckStatus::Pass);
        assert_eq!(report.verdict, Verdict::Experimental);
        assert_eq!(report.dimensions.qualification, Qualification::Experimental);
    }

    #[test]
    fn the_rows_arm_renders_a_source_refusal_as_one_failed_check() {
        let report = super::rows_arm(
            Path::new("/m"),
            Path::new("/t"),
            None,
            rows_stub_refused,
            &Candidate::default(),
            &cuda_facts(24 << 30),
            Qualification::Experimental,
            Vec::new(),
        );
        assert!(report.decisions.is_empty());
        let check = check_named(&report, "load plan");
        assert_eq!(check.status, CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "the load would refuse this pack: the stub refuses"
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
    }

    #[test]
    fn the_flash_family_routes_to_the_rows_arm() {
        let dir = tempfile::tempdir().unwrap();
        let tok = tempfile::tempdir().unwrap();
        let candidate = Candidate {
            family: Some(crate::family::flash_next::factory::FAMILY.to_string()),
            ..Candidate::default()
        };
        let mut facts = cuda_facts(24 << 30);
        facts.host.mem_total_bytes = crate::capability::Probe::Measured(64 << 30);
        facts.host.mem_available_bytes = crate::capability::Probe::Measured(32 << 30);
        let report = super::preflight(dir.path(), tok.path(), &candidate, &facts)
            .expect("an empty directory is inspectable");
        let check = check_named(&report, "load plan");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(
            check.detail.contains("the load would refuse this pack:"),
            "{}",
            check.detail
        );
        assert!(report.decisions.is_empty());
        assert_eq!(report.verdict, Verdict::Unsupported);
    }

    #[test]
    fn the_rows_arm_warns_when_the_fit_is_constrained() {
        let report = super::rows_arm(
            Path::new("/m"),
            Path::new("/t"),
            Some("a family".to_string()),
            rows_stub_constrained,
            &Candidate::default(),
            &cuda_facts(24 << 30),
            Qualification::Experimental,
            Vec::new(),
        );
        let check = check_named(&report, "load plan");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(
            check
                .detail
                .contains("the requested 262144 positions do not fit")
                && check.detail.contains("largest fitting context is 97280")
                && check.detail.contains("a multiple of 1024")
                && check.detail.contains("a stub note"),
            "{}",
            check.detail
        );
        assert_eq!(report.verdict, Verdict::ResourceConstrained);
        assert_eq!(
            report.dimensions.resource_fit,
            super::ResourceFit::DoesNotFit
        );
        assert!(report.next_step.contains("97280"), "{}", report.next_step);
        // The constrained verdict is restated, never stored as an estimate.
        assert!(report.estimate.is_none());
    }

    #[test]
    fn the_rows_arm_warns_instead_of_failing_when_the_probe_missed_a_fact() {
        let report = super::rows_arm(
            Path::new("/m"),
            Path::new("/t"),
            Some("a family".to_string()),
            rows_stub_no_facts,
            &Candidate::default(),
            &cuda_facts(24 << 30),
            Qualification::Experimental,
            Vec::new(),
        );
        let check = check_named(&report, "load plan");
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(
            check
                .detail
                .contains("no decision rows: the device's free memory could not be probed"),
            "{}",
            check.detail
        );
        // A fact the probe could not read is not the pack's fault.
        // A fact the probe could not read leaves the pack experimental,
        // not unsupported: nothing about the pack failed.
        assert_eq!(report.verdict, Verdict::Experimental);
        assert!(report.decisions.is_empty());
        assert_eq!(report.dimensions.resource_fit, super::ResourceFit::Unknown);
    }

    #[test]
    fn a_flash_build_without_the_rows_entry_keeps_the_shallow_report() {
        let dir = tempfile::tempdir().unwrap();
        let tok = tempfile::tempdir().unwrap();
        let inspection =
            crate::capability::inspect::inspect(dir.path(), tok.path()).expect("inspectable");
        let report = super::rows_or_shallow(
            dir.path(),
            tok.path(),
            &inspection,
            Some(crate::family::flash_next::factory::FAMILY.to_string()),
            &Candidate::default(),
            &cuda_facts(24 << 30),
            Vec::new(),
            // The entry a build without the Flash-Next CUDA load path
            // carries: `None`, and the dispatch answers with the shallow
            // report.
            None,
        );
        // With no entry, the shallow report.
        assert!(
            report.checks.iter().any(|c| c.name == "memory estimate"),
            "{:?}",
            report.checks
        );
        assert!(
            report.checks.iter().all(|c| c.name != "load plan"),
            "{:?}",
            report.checks
        );
        assert!(report.decisions.is_empty());
    }

    fn cuda_facts(free_vram: u64) -> super::super::probe::HardwareFacts {
        use crate::capability::Probe;
        use crate::capability::probe::{CompiledBackends, CudaFacts, HardwareFacts, HostFacts};
        HardwareFacts {
            backends: CompiledBackends {
                cuda: true,
                mlx: false,
            },
            cuda: CudaFacts {
                device_count: Probe::Measured(1),
                first_device_integrated: Probe::Measured(false),
                first_device_free_bytes: Probe::Measured(free_vram),
                first_device_total_bytes: Probe::Measured(24u64 << 30),
            },
            host: HostFacts {
                mem_total_bytes: Probe::Unavailable("test".into()),
                mem_available_bytes: Probe::Unavailable("test".into()),
                cpu_threads: 1,
            },
        }
    }

    /// The estimate and the rows each call the one search, so on a real
    /// pack the row's proposal and the estimate's largest fitting context
    /// are the same value. The free bytes sit under the pack's demand so
    /// the row carries a proposal at all.
    /// The row's proposal equals the estimate's value, on a synthetic
    /// plan: the entry's own composition past the walk (rows_from_walk)
    /// runs against the estimate on the same settings, so a build without
    /// a real pack protects the equality.
    #[test]
    fn the_rows_proposal_equals_the_estimates_value_on_a_synthetic_fixture() {
        use crate::family::exl3_pack::PackPlan;
        use crate::family::plan_rows::RowsFit;
        use crate::family::qwen36::preflight::rows_from_walk;
        use crate::family::qwen36::settings::{
            Backend as Q36Backend, Q36Env, Q36Options, Q36Settings,
        };
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir().unwrap();
        let options = Q36Options {
            context_limit: Some(262_144),
            weight_format: Some("exl3".to_string()),
            ..Q36Options::default()
        };
        let settings = Q36Settings::new(
            dir.path(),
            dir.path(),
            Some(&options),
            &Q36Env::default(),
            Q36Backend::Cuda,
        )
        .expect("the pinned spec resolves without a pack");
        let plan = PackPlan {
            device_bytes: 21_000_000_000,
            expert_bytes: 20_000_000_000,
            matrices: 1_976,
            rates: BTreeMap::new(),
            codebook: "Mcg",
        };
        let facts = cuda_facts(23u64 << 30);
        let estimate = super::estimate_q36(
            &settings.spec,
            &settings,
            plan.device_bytes,
            Q36Backend::Cuda,
            &facts,
        )
        .expect("the estimate");
        let (rows, fit) = rows_from_walk(
            &settings,
            &plan,
            20_000_000_000,
            &facts.cuda.first_device_free_bytes,
        )
        .expect("the rows");
        let context = rows
            .iter()
            .find(|decision| decision.name == "context")
            .expect("context row");
        match estimate.largest_fitting_context {
            Some(largest) => {
                assert_eq!(context.alternatives.len(), 1);
                assert_eq!(
                    context.alternatives[0].option,
                    format!("{largest} positions")
                );
                assert!(matches!(
                    fit,
                    RowsFit::Constrained {
                        requested: 262_144,
                        largest: largest_at
                    } if largest_at == largest
                ));
            }
            None => panic!(
                "the fixture free bytes must demand a proposal; an empty equality proves nothing"
            ),
        }
    }

    /// The same equality on the real pack: the full entry (settings,
    /// walk, inventory, rows) against the estimate.
    #[test]
    #[ignore = "reads a real pack in FIFI_EXL3_CHECKPOINT"]
    fn the_rows_proposal_equals_the_estimates_value_on_one_fixture() {
        use crate::family::plan_rows::RowsRequest;
        use crate::family::qwen36::settings::{
            Backend as Q36Backend, Q36Env, Q36Options, Q36Settings,
        };

        let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
        let dir = std::path::PathBuf::from(dir);
        let facts = cuda_facts(23u64 << 30);
        let options = Q36Options {
            context_limit: Some(262_144),
            weight_format: Some("exl3".to_string()),
            ..Q36Options::default()
        };
        let settings = Q36Settings::new(
            &dir,
            &dir,
            Some(&options),
            &Q36Env::default(),
            Q36Backend::Cuda,
        )
        .expect("the pack's settings resolve");
        let plan = crate::family::qwen36::exl3_plan(&dir).expect("the pack walks");
        let estimate = super::estimate_q36(
            &settings.spec,
            &settings,
            plan.device_bytes,
            Q36Backend::Cuda,
            &facts,
        )
        .expect("the estimate");
        let (rows, _note, _fit) = crate::family::qwen36::preflight::preflight_rows(&RowsRequest {
            model_dir: &dir,
            tokenizer_dir: &dir,
            context: Some(262_144),
            kv_dtype: None,
            prefill_rows: None,
            facts: &facts,
        })
        .expect("the rows");
        let context = rows
            .iter()
            .find(|decision| decision.name == "context")
            .expect("context row");
        assert!(
            estimate.largest_fitting_context.is_some(),
            "the fixture free bytes must demand a proposal; an empty equality proves nothing"
        );
        match estimate.largest_fitting_context {
            Some(largest) => {
                assert_eq!(context.alternatives.len(), 1);
                assert_eq!(
                    context.alternatives[0].option,
                    format!("{largest} positions")
                );
            }
            None => assert!(context.alternatives.is_empty()),
        }
    }

    fn check_named<'a>(report: &'a super::PreflightReport, name: &str) -> &'a super::Check {
        report
            .checks
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no check named {name} in {:?}", report.checks))
    }

    #[test]
    fn inkling_mainline_gguf_reports_fixed_loop_without_safetensors() {
        fn key(bytes: &mut Vec<u8>, name: &str, kind: u32) {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(kind.to_le_bytes());
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Inkling-2-Mini-UD-Q4_K_M.gguf_file");
        let numbers = [
            ("block_count", 24u32),
            ("loop.start", 6),
            ("loop.end", 18),
            ("loop.count", 3),
            ("loop.gate_rank", 64),
            ("embedding_length", 2304),
            ("context_length", 1_048_576),
            ("expert_feed_forward_length", 1152),
            ("expert_count", 128),
            ("expert_used_count", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("attention.key_length_swa", 128),
            ("attention.value_length_swa", 128),
            ("attention.sliding_window", 1024),
        ];
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend((numbers.len() as u64 + 4).to_le_bytes());
        key(&mut header, "general.architecture", 8);
        let architecture = b"inkling2_loop";
        header.extend((architecture.len() as u64).to_le_bytes());
        header.extend(architecture);

        for (name, value) in numbers {
            key(&mut header, &format!("inkling2_loop.{name}"), 4);
            header.extend(value.to_le_bytes());
        }
        for (name, count) in [("head_count", 24i32), ("head_count_kv", 8i32)] {
            key(&mut header, &format!("inkling2_loop.attention.{name}"), 9);
            header.extend(5u32.to_le_bytes());
            header.extend(24u64.to_le_bytes());
            for _ in 0..24 {
                header.extend(count.to_le_bytes());
            }
        }
        key(
            &mut header,
            "inkling2_loop.attention.sliding_window_pattern",
            9,
        );
        header.extend(7u32.to_le_bytes());
        header.extend(24u64.to_le_bytes());
        let pattern_start = header.len();
        header.extend((0..24).map(|layer| u8::from(layer % 6 != 5)));

        // A single F32 tensor suffices to exercise inventory and range validation.
        let tensor = b"output_norm.weight";
        header.extend((tensor.len() as u64).to_le_bytes());
        header.extend(tensor);
        header.extend(1u32.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend(0u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.resize(header.len().next_multiple_of(32) + 4, 0);
        std::fs::write(&path, &header).unwrap();

        let run = || {
            super::preflight(
                dir.path(),
                dir.path(),
                &super::Candidate::default(),
                &cuda_facts(24u64 << 30),
            )
            .unwrap()
        };
        let report = run();
        assert_eq!(report.family.as_deref(), Some("inkling2_mini"));
        assert_eq!(report.verdict, Verdict::Unsupported);
        let inventory = check_named(&report, "gguf header");
        assert_eq!(inventory.status, CheckStatus::Pass);
        assert!(
            inventory
                .detail
                .contains("1 tensors, architecture inkling2_loop")
        );
        let layout = check_named(&report, "Inkling 2 Mini GGUF layout");
        assert_eq!(layout.status, CheckStatus::Pass);
        assert!(
            layout
                .detail
                .contains("24 physical layers, 48 layer visits (8 global, 40 sliding-window)")
        );
        assert_eq!(
            check_named(&report, "text execution").detail,
            "Inkling 2 Mini GGUF serving is unsupported"
        );
        assert!(
            !report
                .checks
                .iter()
                .any(|check| check.name == "Inkling 2 Mini expert layout")
        );

        header[pattern_start] = 0;
        std::fs::write(&path, &header).unwrap();
        let invalid = run();
        let layout = check_named(&invalid, "Inkling 2 Mini GGUF layout");
        assert_eq!(layout.status, CheckStatus::Fail);
        assert!(layout.detail.contains("sliding_window_pattern[0]"));
    }

    #[test]
    fn a_wrong_tokenizer_fails_the_pin_check() {
        use crate::family::qwen36::settings::Backend;
        let dir = tempfile::tempdir().unwrap().keep();
        std::fs::write(dir.join("weights.safetensors"), vec![0u8; 8]).unwrap();
        for name in crate::capability::inspect::TOKENIZER_FILES {
            std::fs::write(dir.join(name), "{}").unwrap();
        }
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &super::Candidate::default(),
            &cuda_facts(24u64 << 30),
            Some(Backend::Cuda),
            Vec::new(),
        );
        assert_eq!(
            check_named(&report, "tokenizer pin").status,
            CheckStatus::Fail
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
        assert_eq!(report.dimensions.compatibility, Compatibility::Unsupported);
    }

    /// The MLX arm reads a safetensors pack: the unset `weight_format` (`gguf`) must not demand a
    /// `.gguf` file there, and a directory without any weight file still fails the start.
    #[test]
    fn an_mlx_pack_passes_the_weight_format_check_without_a_gguf_file() {
        use crate::family::qwen36::settings::Backend;
        let candidate = super::Candidate {
            context_limit: Some(4096),
            ..super::Candidate::default()
        };
        let report_for = |with_shard: bool| {
            let dir = tempfile::tempdir().unwrap().keep();
            if with_shard {
                std::fs::write(dir.join("model-00001-of-00001.safetensors"), vec![0u8; 8]).unwrap();
            }
            for name in crate::capability::inspect::TOKENIZER_FILES {
                std::fs::write(dir.join(name), "{}").unwrap();
            }
            let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
            super::q36_preflight(
                &dir,
                &dir,
                &inspection,
                &candidate,
                &cuda_facts(24u64 << 30),
                Some(Backend::Mlx),
                Vec::new(),
            )
        };
        let with = report_for(true);
        let check = check_named(&with, "weight format");
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.detail);
        assert!(
            check.detail.contains("1 MLX safetensors shards"),
            "{}",
            check.detail
        );
        // No weight file: the host facts need the shard bytes, so the settings refuse first.
        let without = report_for(false);
        let check = check_named(&without, "settings");
        assert_eq!(check.status, CheckStatus::Fail, "{}", check.detail);
    }

    #[test]
    fn a_refused_setting_fails_before_any_estimate() {
        use crate::family::qwen36::settings::Backend;
        let dir = tempfile::tempdir().unwrap().keep();
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let candidate = super::Candidate {
            kv_dtype: Some("q4".to_string()),
            ..super::Candidate::default()
        };
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &candidate,
            &cuda_facts(24u64 << 30),
            Some(Backend::Cuda),
            Vec::new(),
        );
        assert_eq!(check_named(&report, "settings").status, CheckStatus::Fail);
        assert!(report.estimate.is_none());
        assert_eq!(report.dimensions.resource_fit, ResourceFit::Unknown);
    }

    #[test]
    fn flint_without_a_cuda_build_is_unsupported() {
        let dir = tempfile::tempdir().unwrap().keep();
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let report = super::flint_preflight(
            &dir,
            &dir,
            &inspection,
            &super::Candidate::default(),
            &cuda_facts(24 << 30),
            false,
            Vec::new(),
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
        assert_eq!(report.dimensions.qualification, Qualification::Unverified);
        assert!(report.decisions.is_empty());
    }

    /// The estimate composes the loader's own placement from the same
    /// pure functions: the bank formula behind the 131072/f16 receipt
    /// (doc/explanation/status.md: 226 of 256 slots on a 24 GB card) is
    /// `floor((free - arena - fixed) / per_slot)` capped at 256; one byte
    /// less is one slot less.
    #[test]
    fn flint_estimate_predicts_the_bank_exactly_and_names_the_alternative() {
        use crate::family::flint::budget as flint_budget;
        use crate::family::flint::exl3_resident::AuditPlan;
        use crate::family::flint::settings::FlintOptions;

        // Twelve routed layers at 2 MiB a slot: per_slot(0) is 24 MiB, and
        // offloading 8 of them leaves 8 MiB — both figures tiny enough that
        // the free-bytes fixtures below stay readable.
        let audit = AuditPlan {
            arena_base: 1 << 30,
            strides: vec![2 << 20; 12],
            routed_layers: 12,
            resident_bytes: 7 << 30,
        };
        let per_slot = audit.per_slot(0).unwrap();
        let options = FlintOptions::cuda_default();
        let env = flint_budget::FlintEnv::default();
        let plan = options.plan().unwrap();
        let (k_row, v_row) = plan.per_plane_row_bytes().unwrap();
        let kv = plan.bytes(k_row + v_row).unwrap() as u64;
        let work = flint_budget::work_reserve_closed_form(
            options.prefill_chunk_size,
            flint_budget::alloc_cache_reserve(env.alloc_cache_enabled()),
        )
        .unwrap();
        let fixed =
            kv + work + flint_budget::FLINT_HEADROOM_BYTES + flint_budget::BANK_MARGIN_BYTES;

        // Free bytes that cover the arena, the fixed charges and exactly
        // 226 slots: the estimate must bank 226, land on the budget line
        // (inside the measurement band), and name the resident ceiling.
        let free = audit.arena_base + fixed + 226 * per_slot;
        let (estimate, decisions) =
            super::flint_estimate(&audit, &options, &env, &cuda_facts(free)).unwrap();
        let placement = decisions
            .iter()
            .find(|decision| decision.name == "expert_placement")
            .expect("expert placement decision");
        assert!(
            placement.chosen.contains("bank: 226 of 256"),
            "{}",
            placement.chosen
        );
        assert!(
            placement.chosen.contains("pinned host memory"),
            "{}",
            placement.chosen
        );
        assert!(estimate.fit == ResourceFit::Unknown && estimate.borderline);
        assert_eq!(estimate.weight_bytes, audit.arena_base + 226 * per_slot);
        let resident_alternative = placement
            .alternatives
            .iter()
            .find(|alternative| alternative.option == "resident")
            .expect("resident alternative");
        assert!(
            resident_alternative
                .reason
                .contains("all-resident would admit at most"),
            "{}",
            resident_alternative.reason
        );
        let largest = estimate
            .largest_fitting_context
            .expect("the banked load proposes the resident ceiling");
        assert!(largest >= 1024 && largest < options.context_limit);
        assert_eq!(largest % 1024, 0);

        // One byte less is one slot less; the refusal side of the same
        // formula, not a different rule.
        let (estimate, _) =
            super::flint_estimate(&audit, &options, &env, &cuda_facts(free - 1)).unwrap();
        assert_eq!(estimate.weight_bytes, audit.arena_base + 225 * per_slot);

        // The offload rule mirrors the loader exactly: CPU-expert layers
        // keep no device slots (per-slot is the sum over the rest), and a
        // huge free device still lands at 255 — never a full bank beside
        // CPU experts — with the lowering named in the decision.
        let offloaded = 8;
        let per_slot_after = audit.per_slot(offloaded).unwrap();
        let mut offload_options = FlintOptions::cuda_default();
        offload_options.cpu_moe_offload_layers = offloaded as u32;
        let free = audit.arena_base + fixed + 200 * per_slot_after;
        let (estimate, _) =
            super::flint_estimate(&audit, &offload_options, &env, &cuda_facts(free)).unwrap();
        assert_eq!(
            estimate.weight_bytes,
            audit.arena_base + 200 * per_slot_after
        );
        let (estimate, decisions) =
            super::flint_estimate(&audit, &offload_options, &env, &cuda_facts(1 << 44)).unwrap();
        let placement = decisions
            .iter()
            .find(|decision| decision.name == "expert_placement")
            .expect("expert placement decision");
        assert!(
            placement.chosen.contains("bank: 255 of 256"),
            "{}",
            placement.chosen
        );
        assert!(
            placement.chosen.contains("lowered from 256"),
            "{}",
            placement.chosen
        );
        assert_eq!(
            estimate.weight_bytes,
            audit.arena_base + 255 * per_slot_after
        );

        // Free bytes that cover the all-resident plan keep every expert
        // on the GPU: the auto decision's other arm.
        let band = crate::family::admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES;
        let resident_free =
            audit.resident_bytes + kv + work + flint_budget::FLINT_HEADROOM_BYTES + band + 1;
        let (estimate, decisions) =
            super::flint_estimate(&audit, &options, &env, &cuda_facts(resident_free)).unwrap();
        let placement = decisions
            .iter()
            .find(|decision| decision.name == "expert_placement")
            .expect("expert placement decision");
        assert!(
            placement.chosen.contains("resident"),
            "{}",
            placement.chosen
        );
        assert_eq!(estimate.weight_bytes, audit.resident_bytes);
        assert_eq!(estimate.fit, ResourceFit::Fits);
        assert!(estimate.largest_fitting_context.is_none());
    }

    #[test]
    fn no_compiled_backend_is_unsupported_not_unverified() {
        let dir = tempfile::tempdir().unwrap().keep();
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &super::Candidate::default(),
            &cuda_facts(0),
            None,
            Vec::new(),
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
        assert_eq!(report.dimensions.qualification, Qualification::Unverified);
    }

    #[test]
    fn dimensions_keep_resource_failure_out_of_compatibility() {
        let dims = q36_dimensions(
            &[Check::new(
                "memory estimate",
                CheckStatus::Fail,
                String::new(),
            )],
            Some(&estimate(2000, Some(1000), Some(90))),
            Qualification::Qualified,
        );
        assert_eq!(dims.compatibility, Compatibility::Supported);
        assert_eq!(dims.resource_fit, ResourceFit::DoesNotFit);
        let dims = q36_dimensions(
            &[Check::new(
                "tokenizer pin",
                CheckStatus::Fail,
                String::new(),
            )],
            None,
            Qualification::Unverified,
        );
        assert_eq!(dims.compatibility, Compatibility::Unsupported);
        assert_eq!(dims.resource_fit, ResourceFit::Unknown);
    }

    #[test]
    fn q36_estimate_uses_the_admission_authorities() {
        use crate::family::qwen36::settings::{Backend, Q36Env, Q36Settings};
        use std::path::Path;

        // Resolved exactly as the factory resolves a bare entry on CUDA.
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            None,
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("default CUDA settings resolve");
        let facts = cuda_facts(24u64 << 30);
        let est = super::estimate_q36(
            &settings.spec,
            &settings,
            20_000_000_000,
            Backend::Cuda,
            &facts,
        )
        .expect("estimate");
        // The CUDA default cache is the F32 control: 40,960 B/position,
        // and the DeltaNet state of the real spec is 65,863,680 bytes —
        // the same figures the settings admission charges by.
        assert_eq!(est.kv_bytes_per_position, 40_960);
        assert_eq!(est.fixed_state_bytes, 65_863_680);
        assert_eq!(est.context_limit, 32_768);
        assert_eq!(
            est.total_bytes,
            20_000_000_000 + 40_960 * 32_768 + 65_863_680
        );
        // 24 GiB free against the full runtime charge model is a confident
        // fit for the 32K F32 default — the case the 850-permille share rule
        // used to refuse even though load-time admission serves it.
        assert_eq!(est.fit, ResourceFit::Fits);
        assert!(!est.borderline);
        assert_eq!(est.largest_fitting_context, None);
    }

    #[test]
    fn the_estimate_matches_the_gpu_qualification_receipts() {
        use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
        use std::path::Path;

        // The 2026-11-03 qualification host: a display-attached RTX 3090,
        // probe free 22.35 GiB, the pinned 19.02 GiB Q4_K_M checkpoint.
        let free = 23_997_186_988u64;
        let weights = 20_419_565_568u64;
        let facts = cuda_facts(free);
        let at = |context: u32, kv: &str| {
            let options: Q36Options = serde_json::from_value(serde_json::json!({
                "context_limit": context,
                "kv_dtype": kv,
            }))
            .expect("options");
            let settings = Q36Settings::new(
                Path::new("m"),
                Path::new("t"),
                Some(&options),
                &Q36Env::default(),
                Backend::Cuda,
            )
            .expect("settings");
            super::estimate_q36(&settings.spec, &settings, weights, Backend::Cuda, &facts)
                .expect("estimate")
        };

        // The 32K F32 default is servable: the old rule called it
        // resource-constrained; the runtime ledger admits it.
        let est = at(32_768, "f32");
        assert_eq!(est.fit, ResourceFit::Fits);
        assert!(!est.borderline);
        assert_eq!(est.largest_fitting_context, None);

        // The shipped 262K Q8 profile on this host: the runtime refused with
        // a 223 MiB shortfall and the estimate must agree, with a real
        // proposal (never the old 0) inside the model's context bound.
        let est = at(262_144, "q8");
        assert_eq!(est.fit, ResourceFit::DoesNotFit);
        assert!(!est.borderline);
        let fitting = est.largest_fitting_context.expect("a proposal");
        assert!(fitting >= 131_072, "{fitting}");
        assert!(fitting < 262_144, "{fitting}");
        assert_eq!(fitting % 1024, 0);
        // The proposal must itself be a confident fit under the same rule.
        let proposed = at(fitting, "q8");
        assert_eq!(proposed.fit, ResourceFit::Fits);

        // An oversized 1M F32 request: refused, but with a usable proposal
        // instead of the degenerate 0 the old division rule produced.
        let est = at(1_048_576, "f32");
        assert_eq!(est.fit, ResourceFit::DoesNotFit);
        let fitting = est.largest_fitting_context.expect("a proposal");
        assert!(fitting >= 60_000, "{fitting}");

        // A free figure landing the 262K Q8 demand inside the measurement
        // band is borderline, not a claim either way.
        let demand_262k_q8 = 3_855_868_032u64;
        let facts = cuda_facts(weights + demand_262k_q8 - (64 << 20));
        let options: Q36Options = serde_json::from_value(serde_json::json!({
            "context_limit": 262_144,
            "kv_dtype": "q8",
        }))
        .expect("options");
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(&options),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("settings");
        let est = super::estimate_q36(&settings.spec, &settings, weights, Backend::Cuda, &facts)
            .expect("estimate");
        assert_eq!(est.fit, ResourceFit::Unknown);
        assert!(est.borderline);
        let (verdict, next) = q36_verdict(&[], Some(&est), Qualification::Qualified);
        assert_eq!(verdict, Verdict::ReadyWithAdjustments);
        assert!(next.contains("measurement band"), "{next}");
    }

    #[test]
    fn servable_excludes_constrained_and_unsupported() {
        assert!(Verdict::Ready.servable());
        assert!(Verdict::Experimental.servable());
        assert!(!Verdict::ResourceConstrained.servable());
        assert!(!Verdict::Unsupported.servable());
    }

    // The preflight mirror of the loader's GGUF admission: every branch of
    // `gguf_admission_check` without touching process-global state.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn gguf_admission_check_mirrors_the_loader_branches() {
        use super::gguf_admission_check;
        let dir = tempfile::tempdir().expect("temp dir");
        let foreign = dir.path().join("Qwen3.6-35B-A3B-Q5_K_M.gguf");
        std::fs::write(&foreign, b"").expect("write");
        // Switch off, pinned file absent: Fail naming the pinned identity and the switch.
        let off = gguf_admission_check(dir.path(), Some("0"));
        assert_eq!(off.name, "weight format");
        assert_eq!(off.status, super::CheckStatus::Fail);
        assert!(
            off.detail.contains("Qwen3.6-35B-A3B-Q4_K_M.gguf")
                && off.detail.contains("Q36_ALLOW_UNPINNED_GGUF"),
            "{}",
            off.detail
        );
        // Switch unset or on, one foreign file: Pass with the admitted-checkpoint note.
        for raw in [None, Some("1")] {
            let on = gguf_admission_check(dir.path(), raw);
            assert_eq!(on.status, super::CheckStatus::Pass);
            assert!(
                on.detail.contains("Q5_K_M") && on.detail.contains("Q36_ALLOW_UNPINNED_GGUF=0"),
                "{}",
                on.detail
            );
        }
        // A bad spelling: Fail before any selection, naming the refusal.
        let bad = gguf_admission_check(dir.path(), Some("yes"));
        assert_eq!(bad.status, super::CheckStatus::Fail);
        assert!(bad.detail.contains("must be 0 or 1"), "{}", bad.detail);
        // Two candidates: the larger file is the one the loader serves.
        std::fs::write(dir.path().join("Qwen3.6-35B-A3B-Q8_0.gguf"), b"larger").expect("write");
        let many = gguf_admission_check(dir.path(), Some("1"));
        assert_eq!(many.status, super::CheckStatus::Pass);
        assert!(many.detail.contains("Q8_0"), "{}", many.detail);
    }
}
