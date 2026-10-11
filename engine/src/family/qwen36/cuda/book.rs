//! The bookkeeping of the resident state, as pure host code (decision 0047 item 4, packet RD-9b1).
//!
//! [`StateBook`] holds the numbers the device owner must never let drift apart: the committed row
//! count `tokens`, the steps enqueued and not yet committed (`pending`), the [`Health`] of the
//! state, the stream that owns it and whether a start block is still owed. `ResidentState` embeds
//! one and routes every state change through it. Each device operation first calls ONE `begin_*`
//! method of the book that holds all its guards, so the guard sequence of every operation is
//! tested on the CPU without a `CudaContext`.
//!
//! # Rules
//! 1. A step may be enqueued only while `tokens + pending + advance <= capacity` (`ensure_rows`,
//!    RD-4d), the sum before the call is the host mirror `admit_pos` of the device `pos` word,
//!    and the advance is at most `MAX_ADVANCE`. Only a `Valid` state enqueues, and not while a
//!    start block is owed (after a reset, an import or a prefill).
//! 2. `commit(accepted, status)` retires the oldest pending step after the host read its result
//!    pair. **In stage 1 ANY non-zero status bit (1, 2, 4, 8 or an unknown bit, alone or in any
//!    combination) poisons the state** and retires nothing (lane M REPLY-76 Q2; relaxing bits 1
//!    and 2 needs a later rule with evidence and lane M's sign-off). The refusal keeps the status
//!    class of the wrapper: `StepStatus::error()` (`Backend`, precedence 4 > 8 > 1 > 2), not
//!    `BadRequest` text. `accepted` must equal the step's advance; a lower count means the step
//!    already ran and the DeltaNet slot is ahead of `tokens` (no per-row snapshot until RD-7c),
//!    so the typed `Partial` refusal POISONS too: the step cannot be undone.
//! 3. Health is a tri-state. `Poisoned` is entered by a status report, a `Partial` commit, a
//!    failed device operation, a reset or an import in progress, and by `allocate` before its
//!    first reset. Only `finish_overwrite` leaves it, reached from `reset` and from
//!    `import_state`; `import_state` may leave a poisoned state ONLY when it does all of
//!    reset's work (the words block rewritten with `StepParams` and status zeroed, the other
//!    slots cleared, a full overwrite from a snapshot `check_import` accepted); `begin_import`
//!    refuses a `Poisoned` state otherwise (REPLY-76 Q1), and a failure after `check_import`
//!    leaves it `Poisoned`, never a partly imported `Valid` state. `PrefillOpen` is entered by
//!    `begin_prefill` from `Valid` and idle; only `revalidate` leaves it, and only from there. A
//!    mask-4 defect therefore stays until reset: `revalidate` cannot clear it.
//! 4. Export, `revalidate`, `begin_prefill` and a start block need `pending == 0`: a state copied
//!    with work in flight is ahead of `tokens`, imports cleanly and decodes plausible wrong text.
//! 5. Every device copy and every block goes through the stream that owns the state
//!    (`DeviceStream::id`); another stream races the work in flight. `join_branches` is the only
//!    cross-stream path.
//!
//! Design guidance only, no code copied; the strict per-component accounting follows the idea of
//! MegaCapybara C4 and the slot-indexed state is the C12 hook (the MegaCapybara static-RE lessons
//! section 6.1).

use std::fmt;

use crate::{
    EngError,
    kernels::cuda::resident::step_params::StepStatus,
    kernels::cuda::resident::step_params::{MAX_ADVANCE, ensure_rows},
    kernels::cuda::resident::step_params::{POS_MAX, StepParams},
};

/// Whether the state may be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Health {
    Valid,
    /// A prefill chunk is writing the pools in place (decision 0049).
    PrefillOpen,
    /// Untrustworthy until a reset or a successful import.
    Poisoned,
}

/// Steps enqueued and not yet committed, oldest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PendingWork {
    advances: Vec<usize>,
}

/// What `enqueue` hands back: the host mirror of the device `pos` word at the start of the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Admitted {
    pub(super) admit_pos: usize,
}

/// Why a commit was refused. A status report and a `Partial` commit poison the book; the other
/// variants change nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitRefusal {
    /// The state was not `Valid` (the text names the health).
    NotValid,
    /// No step is pending.
    NothingPending,
    /// The result pair carried a non-zero status word; the state is now `Poisoned`.
    Status(i32),
    /// Stage 1 commits whole steps only (module doc, rule 2); the state is now `Poisoned`.
    Partial { advance: usize, accepted: usize },
}

/// The error of a non-zero status word, with the class of the wrapper (`StepStatus::error`,
/// `Backend`): mask 4 `step_params`, then mask 8 `prefill_chunk`, then 1 `route_experts`, then 2
/// `sample` (precedence 4 > 8 > 1 > 2); unknown bits alone are a `Backend` `step_status`.
pub(crate) fn status_error(status: i32) -> EngError {
    let report = StepStatus(status);
    if report.invalid_step_state() {
        return report.error().unwrap_or_else(|| unknown_bits(status));
    }
    if status & 8 != 0 {
        return EngError::Backend {
            operation: "prefill_chunk",
            message: "the prefill chunk reported a fault".to_owned(),
        };
    }
    report.error().unwrap_or_else(|| unknown_bits(status))
}

fn unknown_bits(status: i32) -> EngError {
    EngError::Backend {
        operation: "step_status",
        message: format!("the status word {status} holds bits that no mask names"),
    }
}

impl fmt::Display for CommitRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotValid => write!(f, "the resident state is not valid; reset first"),
            Self::NothingPending => write!(f, "no step is pending, so there is nothing to commit"),
            Self::Status(status) => write!(
                f,
                "the step reported status {status}; the resident state is poisoned until reset"
            ),
            Self::Partial { advance, accepted } => write!(
                f,
                "a step of {advance} rows cannot commit {accepted}: partial accept needs the DeltaNet snapshots of RD-7c, and the state is poisoned"
            ),
        }
    }
}

impl From<CommitRefusal> for EngError {
    fn from(refusal: CommitRefusal) -> Self {
        match refusal {
            CommitRefusal::Status(status) => status_error(status),
            other => EngError::BadRequest(format!("commit_step: {other}")),
        }
    }
}

/// Why a start block was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StartRefusal {
    NotValid,
    NotIdle,
    Token(i32),
    PosRange(i32),
    PosMirror { pos: i32, tokens: usize },
    KvLen { kv_len: i32, pos: i32 },
    KvSlot { kv_slot: i32, pos: i32 },
    Status(i32),
}

impl fmt::Display for StartRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotValid => write!(f, "the resident state is not valid"),
            Self::NotIdle => write!(f, "steps are still pending, so the device owns the words"),
            Self::Token(t) => write!(f, "the start token {t} is negative"),
            Self::PosRange(p) => write!(f, "the start position {p} is outside 0..={POS_MAX}"),
            Self::PosMirror { pos, tokens } => {
                write!(
                    f,
                    "the start position {pos} differs from the {tokens} committed rows"
                )
            }
            Self::KvLen { kv_len, pos } => write!(f, "kv_len {kv_len} is not pos {pos} + 1"),
            Self::KvSlot { kv_slot, pos } => write!(f, "kv_slot {kv_slot} is not pos {pos}"),
            Self::Status(s) => write!(f, "the start status {s} is not 0"),
        }
    }
}

impl From<StartRefusal> for EngError {
    fn from(refusal: StartRefusal) -> Self {
        EngError::BadRequest(format!("write_step_params: {refusal}"))
    }
}

/// The committed count, the pending steps, the health and the owning stream of one state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StateBook {
    capacity: usize,
    tokens: usize,
    pending: PendingWork,
    health: Health,
    owner: u64,
    /// A reset, an import or a prefill leaves the device words without a valid start block; the
    /// host must write one (`write_step_params`) before the next step is enqueued.
    needs_start: bool,
}

fn refuse(what: &str, why: &str) -> EngError {
    EngError::BadRequest(format!("{what}: {why}"))
}

impl StateBook {
    /// A book for a state of `capacity` KV rows owned by stream `owner`. It starts `Poisoned`:
    /// the buffers are uninitialized until the first reset finishes.
    pub(super) fn new(capacity: usize, owner: u64) -> Self {
        Self {
            capacity,
            tokens: 0,
            pending: PendingWork::default(),
            health: Health::Poisoned,
            owner,
            needs_start: true,
        }
    }

    pub(super) fn health(&self) -> Health {
        self.health
    }

    pub(super) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(super) fn pending_rows(&self) -> usize {
        self.pending.advances.iter().sum()
    }

    pub(super) fn needs_start(&self) -> bool {
        self.needs_start
    }

    /// `BadRequest` unless `stream_id` is the owning stream (rule 5).
    fn check_owner(&self, stream_id: u64, what: &str) -> Result<(), EngError> {
        if stream_id == self.owner {
            return Ok(());
        }
        Err(refuse(
            what,
            &format!(
                "the state is owned by stream {}, not stream {stream_id}",
                self.owner
            ),
        ))
    }

    fn require_valid(&self, what: &str) -> Result<(), EngError> {
        if self.health == Health::Valid {
            return Ok(());
        }
        Err(refuse(
            what,
            &format!(
                "the resident state is {:?}; reset or import again",
                self.health
            ),
        ))
    }

    /// `BadRequest` unless nothing is pending (rule 4).
    fn require_idle(&self, what: &str) -> Result<(), EngError> {
        if self.pending.advances.is_empty() {
            return Ok(());
        }
        Err(refuse(
            what,
            &format!(
                "needs a committed boundary, but {} steps ({} rows) are pending",
                self.pending.advances.len(),
                self.pending_rows()
            ),
        ))
    }

    /// Guards of `join_branches`: `main` is the owning stream (a poisoned state may be joined).
    pub(super) fn begin_join(&self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "join_branches")
    }

    /// Guards of `Mixers::run_block`: the owning stream and a valid state.
    pub(super) fn begin_block(&self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "run_block")?;
        self.require_valid("run_block")
    }

    /// Guards of `export_state`: the owning stream, a valid and idle state.
    pub(super) fn begin_export(&self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "export_state")?;
        self.require_valid("export_state")?;
        self.require_idle("export_state")
    }

    /// Guards of `reset`: the owning stream. The state is `Poisoned` until `finish_overwrite`.
    pub(super) fn begin_reset(&mut self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "reset")?;
        self.health = Health::Poisoned;
        Ok(())
    }

    /// Guards of `import_state` after `check_import` accepted the value: the owning stream, and
    /// `rewrites_words` - whether the import does ALL of reset's work (the words block rewritten
    /// with `StepParams` and status zeroed, the other slots cleared). A `Poisoned` state is
    /// refused unless it does (REPLY-76 Q1); a `Valid` or `PrefillOpen` one may be replaced
    /// either way. The state is `Poisoned` until `finish_overwrite`, so a failure after this call
    /// never leaves a partly imported `Valid` state. `import_state` always passes `true`
    /// (`IMPORT_REWRITES_WORDS`), so the `false` refusal below is unreachable in production: it
    /// is a guard for future callers that import without rewriting the words.
    pub(super) fn begin_import(
        &mut self,
        stream_id: u64,
        rewrites_words: bool,
    ) -> Result<(), EngError> {
        self.check_owner(stream_id, "import_state")?;
        if self.health == Health::Poisoned && !rewrites_words {
            return Err(refuse(
                "import_state",
                "a poisoned state is imported only together with reset's work on the words; reset first",
            ));
        }
        self.health = Health::Poisoned;
        Ok(())
    }

    /// Guards of `poison_device`: the owning stream (checked BEFORE the state is poisoned, so a
    /// wrong-stream caller cannot poison it). Poisons.
    pub(super) fn begin_poison_device(&mut self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "poison_device")?;
        self.health = Health::Poisoned;
        Ok(())
    }

    /// A reset or an import finished: `tokens` committed rows, nothing pending, `Valid`, and a
    /// start block owed.
    pub(super) fn finish_overwrite(&mut self, tokens: usize) {
        self.tokens = tokens;
        self.pending.advances.clear();
        self.health = Health::Valid;
        self.needs_start = true;
    }

    /// The host wrote a start block that `begin_write_params` accepted.
    pub(super) fn params_written(&mut self) {
        self.needs_start = false;
    }

    /// Admits a step of `advance` rows (rule 1).
    pub(super) fn enqueue(&mut self, advance: usize) -> Result<Admitted, EngError> {
        self.require_valid("enqueue_rows")?;
        if self.needs_start {
            return Err(refuse(
                "enqueue_rows",
                "the device words hold no start block; write_step_params first",
            ));
        }
        if advance > MAX_ADVANCE {
            return Err(refuse(
                "enqueue_rows",
                &format!("a step advances at most {MAX_ADVANCE} rows, not {advance}"),
            ));
        }
        let admit_pos = self
            .tokens
            .checked_add(self.pending_rows())
            .ok_or_else(|| refuse("enqueue_rows", "pending rows overflow"))?;
        ensure_rows(admit_pos, advance, self.capacity)?;
        self.pending.advances.push(advance);
        Ok(Admitted { admit_pos })
    }

    /// Takes back the newest pending step, for a step whose enqueue is to be undone before any
    /// launch was queued (a failure before `run_block`). After `run_block` returned an error the
    /// caller poisons instead: queued kernels may have run.
    pub(super) fn cancel_last(&mut self) -> Result<(), EngError> {
        self.pending
            .advances
            .pop()
            .map(|_| ())
            .ok_or_else(|| refuse("cancel_enqueue", "no step is pending"))
    }

    /// Commits the oldest pending step after the host read its result pair, whose status word is
    /// `status` (rule 2). Validates before it changes anything, except that a non-zero status and
    /// a `Partial` commit poison.
    pub(super) fn commit(&mut self, accepted: usize, status: i32) -> Result<(), CommitRefusal> {
        if self.health != Health::Valid {
            return Err(CommitRefusal::NotValid);
        }
        if status != 0 {
            self.health = Health::Poisoned;
            return Err(CommitRefusal::Status(status));
        }
        let Some(&advance) = self.pending.advances.first() else {
            return Err(CommitRefusal::NothingPending);
        };
        if accepted != advance {
            self.health = Health::Poisoned;
            return Err(CommitRefusal::Partial { advance, accepted });
        }
        self.pending.advances.remove(0);
        self.tokens += accepted;
        Ok(())
    }

    /// Enters `Poisoned` (rule 3): the caller's verdict that the state cannot be trusted (any
    /// non-zero status bit, any `Err` from `run_block`). Pending steps stay counted until a reset
    /// or an import.
    pub(super) fn poison(&mut self) {
        self.health = Health::Poisoned;
    }

    /// Enters `PrefillOpen` from `Valid` and idle.
    pub(super) fn begin_prefill(&mut self) -> Result<(), EngError> {
        self.require_valid("begin_prefill")?;
        self.require_idle("begin_prefill")?;
        self.health = Health::PrefillOpen;
        Ok(())
    }

    /// Guards of `revalidate`: the owning stream and an open prefill. Checked BEFORE the caller
    /// synchronizes the stream, so a refused call neither waits nor poisons.
    pub(super) fn begin_revalidate(&self, stream_id: u64) -> Result<(), EngError> {
        self.check_owner(stream_id, "revalidate")?;
        self.require_prefill_open()
    }

    /// `BadRequest` unless a prefill is in progress.
    fn require_prefill_open(&self) -> Result<(), EngError> {
        if self.health == Health::PrefillOpen {
            return Ok(());
        }
        Err(refuse(
            "revalidate",
            &format!(
                "only a prefill in progress can be revalidated, the state is {:?}",
                self.health
            ),
        ))
    }

    /// Leaves `PrefillOpen` after a chunk whose prefill status word was `prefill_status` and
    /// publishes `tokens` rows (it may be fewer than before: a prefill that restarts). A
    /// non-zero status poisons. A start block is owed afterwards.
    pub(super) fn revalidate(
        &mut self,
        tokens: usize,
        prefill_status: i32,
    ) -> Result<(), EngError> {
        self.require_prefill_open()?;
        if prefill_status != 0 {
            self.health = Health::Poisoned;
            return Err(status_error(prefill_status));
        }
        self.require_idle("revalidate")?;
        if tokens > self.capacity {
            return Err(EngError::ContextLimit(format!(
                "{tokens} positions exceed the KV capacity {}",
                self.capacity
            )));
        }
        self.tokens = tokens;
        self.health = Health::Valid;
        self.needs_start = true;
        Ok(())
    }

    /// Guards of `write_step_params`: the owning stream, then the start block against the book
    /// (stage 1 decode).
    pub(super) fn begin_write_params(
        &self,
        stream_id: u64,
        params: &StepParams,
    ) -> Result<(), EngError> {
        self.check_owner(stream_id, "write_step_params")?;
        self.check_start(params)?;
        Ok(())
    }

    pub(super) fn check_start(&self, params: &StepParams) -> Result<(), StartRefusal> {
        if self.health != Health::Valid {
            return Err(StartRefusal::NotValid);
        }
        if !self.pending.advances.is_empty() {
            return Err(StartRefusal::NotIdle);
        }
        if params.token_id < 0 {
            return Err(StartRefusal::Token(params.token_id));
        }
        if !(0..=POS_MAX).contains(&params.pos) {
            return Err(StartRefusal::PosRange(params.pos));
        }
        if usize::try_from(params.pos) != Ok(self.tokens) {
            return Err(StartRefusal::PosMirror {
                pos: params.pos,
                tokens: self.tokens,
            });
        }
        if params.kv_len != params.pos + 1 {
            return Err(StartRefusal::KvLen {
                kv_len: params.kv_len,
                pos: params.pos,
            });
        }
        if params.kv_slot != params.pos {
            return Err(StartRefusal::KvSlot {
                kv_slot: params.kv_slot,
                pos: params.pos,
            });
        }
        if params.status != 0 {
            return Err(StartRefusal::Status(params.status));
        }
        Ok(())
    }
}
