//! What one run of the resident step reports beside its token (decision 0047, packet RD-9d; the
//! `captured` flag of RD-10a/RD-10b). Pure host values: no device type, no CUDA call.
//!
//! [`CaptureOutcome`] says how the step body ran. RD-9d only ever runs it eagerly, so the value is
//! always [`CaptureOutcome::Eager`] (the default). The slim RD-11a (lane C) is the packet that
//! captures the step: it sets [`CaptureOutcome::Captured`] or
//! [`CaptureOutcome::RefusedFellBackToEager`], and it widens `is_capture_refusal` to `pub(crate)`
//! for that purpose (lane C, Q13). The RD-10 gate reads `RunReport::capture` of the step's last
//! run (`QwenResidentStep::last_report`) and never calls `is_capture_refusal` itself. Any other
//! capture error is an error of the run, never a fallback.
//!
//! Design guidance only, no code copied (no MegaCapybara-derived design).

/// How the step body of a run was executed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum CaptureOutcome {
    /// Every launch was queued one by one. The only value RD-9d produces.
    #[default]
    Eager,
    /// The body was captured into a graph and the graph was replayed.
    Captured,
    /// A capture was requested and refused with a status on the closed list; the same body then
    /// ran eagerly.
    RefusedFellBackToEager,
}

impl CaptureOutcome {
    /// True only when the graph was captured and replayed: the RD-10 `captured` flag.
    pub(crate) const fn captured(self) -> bool {
        matches!(self, Self::Captured)
    }

    /// The text of the raw gate file.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Captured => "captured",
            Self::RefusedFellBackToEager => "refused_fell_back_to_eager",
        }
    }
}

/// The report of the last run of the step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RunReport {
    /// How the body ran.
    pub(crate) capture: CaptureOutcome,
    /// Host waits for the run to finish, per token (the "syncs per token" of RD-10b).
    pub(crate) syncs: u32,
    /// Blocking host-to-device copies before the launches (the start block of the step).
    pub(crate) host_writes: u32,
}
