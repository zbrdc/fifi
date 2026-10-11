//! How the host reads one token of the resident step back, and the mode switches around it
//! (decision 0047, packet RD-9d). No device type appears here, so the rules are tested on the CPU
//! without a `CudaContext`.
//!
//! # Read-back
//!
//! One host wait per token. The pair copy is queued and an event recorded behind it; the logits
//! row (when wanted) is read next, which waits for the stream; the pair is read last and its
//! event is complete by then. See [`ReadbackPlan`]. The start block of the token is one blocking
//! host-to-device copy before the launches ([`HOST_WRITES_PER_TOKEN`]); the report counts it
//! apart from the wait.
//!
//! # Modes
//!
//! `INFERENCE_CUDA_SYNC_EACH=1` makes the eager step synchronize after every launch, for
//! compute-sanitizer. A synchronize inside a capture window is refused, so the mode and a capture
//! exclude each other.
//!
//! Design guidance only, no code copied; the single host wait per token follows the idea of
//! MegaCapybara C1 (research/megacapybara-re-lessons.md section 6.1), without its ring.

use super::run_report::{CaptureOutcome, RunReport};
use crate::EngError;

/// The environment variable that makes the eager step synchronize after every launch.
pub(super) const SYNC_EACH_VAR: &str = "INFERENCE_CUDA_SYNC_EACH";
/// Blocking host-to-device copies before the launches of a token: the start block.
const HOST_WRITES_PER_TOKEN: u32 = 1;

/// Reads the value of `INFERENCE_CUDA_SYNC_EACH`: unset, empty or `0` is off, `1` is on, any
/// other text is a `BadRequest`.
pub(super) fn parse_sync_each(value: Option<&str>) -> Result<bool, EngError> {
    match value {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR} must be 0 or 1, got {other:?}"
        ))),
    }
}

/// A synchronize inside a capture window is refused by the context, so the debug mode and the
/// captured path exclude each other.
pub(super) fn check_mode(sync_each: bool, capture_requested: bool) -> Result<(), EngError> {
    if sync_each && capture_requested {
        return Err(EngError::BadRequest(format!(
            "{SYNC_EACH_VAR}=1 synchronizes after every launch, which a graph capture forbids"
        )));
    }
    Ok(())
}

/// One host-side action of the read-back, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadStep {
    QueuePairCopy,
    RecordEvent,
    ReadLogits,
    ReadPair,
}

/// What the host reads after the launches of a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReadbackPlan {
    /// Bytes of the logits row, or `None` when the caller does not want it.
    pub(super) logits_bytes: Option<usize>,
}

impl ReadbackPlan {
    /// A plan for a head of `vocab` values (f32) and the caller's choice about the row.
    pub(super) fn new(vocab: usize, want_logits: bool) -> Result<Self, EngError> {
        if vocab == 0 {
            return Err(EngError::BadRequest(
                "the read-back plan needs a vocabulary of at least one value".to_owned(),
            ));
        }
        let logits_bytes = if want_logits {
            let bytes = vocab
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| EngError::BadRequest("the logits row size overflows".to_owned()))?;
            Some(bytes)
        } else {
            None
        };
        Ok(Self { logits_bytes })
    }

    /// The order of the actions. The pair copy is queued and its event recorded before any
    /// blocking read, so the first blocking read waits for everything the token queued.
    pub(super) fn steps(&self) -> Vec<ReadStep> {
        let mut steps = vec![ReadStep::QueuePairCopy, ReadStep::RecordEvent];
        if self.logits_bytes.is_some() {
            steps.push(ReadStep::ReadLogits);
        }
        steps.push(ReadStep::ReadPair);
        steps
    }

    /// Host waits that can block: the first read after the record. The reads behind it find the
    /// work complete.
    pub(super) fn blocking_waits(&self) -> u32 {
        let first_read = self
            .steps()
            .into_iter()
            .position(|step| matches!(step, ReadStep::ReadLogits | ReadStep::ReadPair));
        u32::from(first_read.is_some())
    }
}

/// The report of a token that ran eagerly (RD-9d runs nothing else).
pub(super) fn eager_report(plan: &ReadbackPlan) -> RunReport {
    RunReport {
        capture: CaptureOutcome::default(),
        syncs: plan.blocking_waits(),
        host_writes: HOST_WRITES_PER_TOKEN,
    }
}
