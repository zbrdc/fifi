//! Startup decisions from the loaded MLX executor. These records contain
//! only portable values and can be read without querying Metal.

use std::collections::BTreeMap;

use serde::Serialize;

use super::plan_decision::{PlanAlternative, PlanDecision, PlanFixity};

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
#[cfg_attr(
    not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) enum PrefillReason {
    Admitted,
    Configured,
    NoBudget,
    RowCeiling,
    Workspace,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MlxExecutionPlan {
    pub(crate) backend: &'static str,
    pub(crate) configured_prefill_rows: usize,
    pub(crate) prefill_rows: usize,
    pub(crate) decisions: [PlanDecision; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory: Option<MlxPrefillMemory>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MlxPrefillMemory {
    pub(crate) prefill_budget_bytes: u64,
    pub(crate) remaining_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) device: Option<MlxDeviceCeiling>,
    pub(crate) base_charges: BTreeMap<&'static str, u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) workspace: BTreeMap<&'static str, u64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct MlxDeviceCeiling {
    pub(crate) metal_working_set_bytes: u64,
    pub(crate) effective_device_budget_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) configured_device_ceiling_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LoadedMlxExecution {
    #[serde(flatten)]
    pub(crate) plan: MlxExecutionPlan,
    pub(crate) worker_piece_rows: usize,
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
impl MlxExecutionPlan {
    pub(crate) fn new(
        placement: &'static str,
        placement_reason: &'static str,
        configured_rows: usize,
        selected_rows: usize,
        reason: PrefillReason,
        memory: Option<MlxPrefillMemory>,
    ) -> Self {
        let prefill = if selected_rows == 1 {
            "one_token"
        } else {
            "chunked"
        };
        let explanation = match reason {
            PrefillReason::Admitted => "chunk fits the workspace budget",
            PrefillReason::Configured => "using the configured prefill width",
            PrefillReason::NoBudget => "no prefill budget was configured",
            PrefillReason::RowCeiling => "configured width is below the minimum chunk",
            PrefillReason::Workspace => "no supported chunk fits the workspace budget",
        };
        let decision = |name: &str, chosen: &str, reason: &str| PlanDecision {
            name: name.into(),
            chosen: chosen.into(),
            alternatives: vec![PlanAlternative {
                option: chosen.into(),
                verdict: "chosen".into(),
                reason: reason.into(),
            }],
            fixed_at: PlanFixity::Load,
        };
        Self {
            backend: "mlx",
            configured_prefill_rows: configured_rows,
            prefill_rows: selected_rows,
            decisions: [
                decision("weight_placement", placement, placement_reason),
                decision("prefill", prefill, explanation),
            ],
            memory,
        }
    }
}
