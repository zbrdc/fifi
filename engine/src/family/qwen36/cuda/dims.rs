//! The checked dimensions of one resident step and the options fixed for a session (decision
//! 0047 item 3, packet RD-9a).
//!
//! Pure host arithmetic. Geometry comes from the layout record ([`HostShape`], lane M's
//! `family/qwen36`); the widths of the fused projection outputs come from the output rows of the
//! tensor roles in the inventory (lane C), checked against the geometry. No model number is
//! written here.

use super::super::host::HostShape;
use crate::{
    EngError,
    family::{
        AttentionKind,
        gguf_model::{Inventory, Role},
        project::cuda::WeightLayout,
    },
};

/// How the KV cache stores K and V. Fixed for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvMode {
    Fp16,
    Int8G32,
}

/// What the server asks for; `Auto` is resolved once, at session admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvRequest {
    Fp16,
    Int8,
    Auto,
}

impl KvRequest {
    /// `Auto` takes `Fp16` when its KV bytes fit `budget`, else `Int8G32`. `budget` is the frozen
    /// headroom of the host (decision 0047 A2), passed by the admission caller; there is no
    /// default number here, and `Auto` without a frozen budget is an error, not a guess
    /// (REPLY-43 item 6).
    pub(crate) fn resolve(
        self,
        fp16_bytes: usize,
        budget: Option<usize>,
    ) -> Result<KvMode, EngError> {
        match (self, budget) {
            (Self::Fp16, _) => Ok(KvMode::Fp16),
            (Self::Int8, _) => Ok(KvMode::Int8G32),
            (Self::Auto, Some(budget)) if fp16_bytes <= budget => Ok(KvMode::Fp16),
            (Self::Auto, Some(_)) => Ok(KvMode::Int8G32),
            (Self::Auto, None) => Err(EngError::BadRequest(
                "KV mode auto needs the frozen headroom budget of this host".to_owned(),
            )),
        }
    }
}

/// Options fixed for one session (the KV mode cannot change inside it: KV addresses are baked
/// into captured graphs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepOptions {
    pub(crate) kv_mode: KvMode,
    /// Positions the KV cache holds; allocated once, never grown.
    pub(crate) context_tokens: usize,
    /// DeltaNet state slots; slot 0 is the committed state (stage 1 uses one). A verify window of
    /// M rows (MEL-PLAN 6.5) may need one more slot per row for rollback; that count is RD-7c's
    /// decision, and every slot costs one `DeltaPool` slot (62.8125 MiB on Qwen3.6-35B-A3B).
    pub(crate) delta_slots: usize,
    /// Stage 1b: one `xf` buffer per routed expert, so the experts can run on branch streams.
    pub(crate) forked_experts: bool,
    /// The device layout the step runs (MATVEC-PAR S3b): the q8_1 producers and MMVQ matvecs of
    /// `Native`, or today's fp16 packs and tiled matmuls of `Tiled`. DERIVED from
    /// `CudaWeights::layout()` of the loaded model by every production caller (the gate); the
    /// step refuses a mismatch of the two, and the default keeps every caller on the tiled path.
    pub(crate) layout: WeightLayout,
    /// The tiered expert mode (OFF-2): `None` is the plain native step. Set only by an explicit
    /// config key; the step then runs eager with a host decision in every layer (decision 0047,
    /// amendment of 2026-10-13).
    pub(crate) tiered: Option<TieredPlan>,
}

/// The numbers of the tiered expert mode, plain so `StepOptions` stays `Copy` and `Eq`
/// (fractions are milli-units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TieredPlan {
    /// Bank slots per layer; at least the picks per token, at most the experts of a layer.
    pub(crate) slots_per_layer: usize,
    /// CPU worker threads, 1 to [`TieredPlan::MAX_THREADS`].
    pub(crate) threads: usize,
    /// The longest the host waits for one layer's CPU experts.
    pub(crate) timeout_ms: u32,
    /// Bank swaps per token in thousandths (250 is 0.25).
    pub(crate) swaps_milli: u32,
    /// Popularity decay per token in thousandths (999 is 0.999).
    pub(crate) decay_milli: u32,
    /// Fill the banks with the lowest-numbered experts of every layer before the first token. With
    /// a bank as deep as the layer it makes every pick resident, which the bitwise gate against the
    /// plain step needs.
    pub(crate) preload: bool,
}

impl TieredPlan {
    /// The thread cap (git-db, 2026-10-13: the machine is on its thermal limit).
    pub(crate) const MAX_THREADS: usize = 8;

    /// The defaults of the measurements: the caller chooses the slots and the thread count (6 in
    /// the measurements); timeout 10 s, 0.25 swaps per token, decay 0.999, no preload.
    pub(crate) fn new(slots_per_layer: usize, threads: usize) -> Self {
        Self {
            slots_per_layer,
            threads,
            timeout_ms: 10_000,
            swaps_milli: 250,
            decay_milli: 999,
            preload: false,
        }
    }

    /// Refuses a plan the host or the policy cannot run, naming the field.
    pub(crate) fn check(&self) -> Result<(), EngError> {
        let bad = |what: String| Err(EngError::BadRequest(format!("tiered plan: {what}")));
        if self.slots_per_layer == 0 {
            return bad(String::from("slots_per_layer must be above zero"));
        }
        if !(1..=Self::MAX_THREADS).contains(&self.threads) {
            return bad(format!(
                "threads {} is outside 1..={}",
                self.threads,
                Self::MAX_THREADS
            ));
        }
        if !(1..=600_000).contains(&self.timeout_ms) {
            return bad(format!(
                "timeout_ms {} is outside 1..=600000",
                self.timeout_ms
            ));
        }
        if self.swaps_milli > 8_000 {
            return bad(format!("swaps_milli {} is above 8000", self.swaps_milli));
        }
        if !(1..=1_000).contains(&self.decay_milli) {
            return bad(format!(
                "decay_milli {} is outside 1..=1000",
                self.decay_milli
            ));
        }
        Ok(())
    }
}

/// The checked dimensions of the step: the layout record plus the widths of the fused
/// projection outputs, read from the inventory.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StepDims {
    pub(crate) shape: HostShape,
    /// Rows of `AttnQkv`, `AttnGate`, `SsmBeta` and `SsmAlpha` together.
    pub(super) delta_proj: usize,
    /// Rows of `AttnQ` (query and output gate), `AttnK` and `AttnV` together.
    pub(super) attn_proj: usize,
}

impl StepDims {
    /// Reads the projection widths from `inventory` and checks them against `shape`.
    pub(crate) fn new(shape: &HostShape, inventory: &Inventory) -> Result<Self, EngError> {
        shape.check()?;
        let d = &shape.delta;
        let a = &shape.attn;
        let m = &shape.moe;
        let q_width = a.heads_q * a.head_dim;
        let kv_width = a.heads_kv * a.head_dim;
        let checks = [
            (Role::AttnQkv, d.conv_dim()),
            (Role::AttnGate, d.value_heads * d.value_dim),
            (Role::SsmBeta, d.value_heads),
            (Role::SsmAlpha, d.value_heads),
            (Role::SsmOut, shape.hidden),
            (Role::AttnQ, 2 * q_width),
            (Role::AttnK, kv_width),
            (Role::AttnV, kv_width),
            (Role::AttnOutput, shape.hidden),
            (Role::FfnGateInp, m.experts),
            (Role::FfnGateInpShexp, 1),
            (Role::FfnGateExps, m.inter),
            (Role::FfnUpExps, m.inter),
            (Role::FfnDownExps, shape.hidden),
            (Role::FfnGateShexp, m.shared_inter),
            (Role::FfnUpShexp, m.shared_inter),
            (Role::FfnDownShexp, shape.hidden),
            (Role::Output, shape.vocab),
        ];
        for (role, expected) in checks {
            let rows = role_rows(inventory, role)?;
            if rows != expected {
                return Err(EngError::Checkpoint(format!(
                    "step plan: role {role:?} has {rows} output rows, the layout record needs {expected}"
                )));
            }
        }
        let rows = |role| role_rows(inventory, role);
        Ok(Self {
            shape: shape.clone(),
            delta_proj: rows(Role::AttnQkv)? + rows(Role::AttnGate)? + 2 * rows(Role::SsmBeta)?,
            attn_proj: rows(Role::AttnQ)? + rows(Role::AttnK)? + rows(Role::AttnV)?,
        })
    }

    pub(crate) fn layers_of(&self, kind: AttentionKind) -> usize {
        self.shape.kinds.iter().filter(|&&k| k == kind).count()
    }
}

/// Output rows of the tensor of `role`: `dims[1]` of its GGUF dims, 1 for a vector. Every layer
/// that holds the role must agree.
fn role_rows(inventory: &Inventory, role: Role) -> Result<usize, EngError> {
    let mut found: Option<u64> = None;
    for tensor in inventory.tensors().iter().filter(|t| t.role == role) {
        let rows = tensor.dims.get(1).copied().unwrap_or(1);
        if found.is_some_and(|earlier| earlier != rows) {
            return Err(EngError::Checkpoint(format!(
                "step plan: tensor {} gives role {role:?} another row count",
                tensor.name
            )));
        }
        found = Some(rows);
    }
    found
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(|| EngError::Checkpoint(format!("step plan: the inventory has no {role:?}")))
}
