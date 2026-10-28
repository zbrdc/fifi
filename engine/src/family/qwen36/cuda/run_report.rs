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

use crate::{family::session::PrefixSlot, sched::job::PrefixReport};

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

/// The host waits the tiered mode adds to a token: each one blocks the host until the stream has
/// reached it (an event wait, or a blocking host-to-device write, which returns when the copy
/// finished behind the queued work). A plain step adds none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct HostSyncs {
    /// Event waits.
    pub(super) waits: u32,
    /// Blocking host-to-device writes.
    pub(super) writes: u32,
}

impl HostSyncs {
    /// Adds the counts of `other` into `self`.
    pub(super) fn add(&mut self, other: HostSyncs) {
        self.waits += other.waits;
        self.writes += other.writes;
    }
}

/// The report of the last run of the step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RunReport {
    /// How the body ran.
    pub(crate) capture: CaptureOutcome,
    /// Host waits for the run to finish, per token (the "syncs per token" of RD-10b). A plain step
    /// makes one. A tiered token adds the waits of every layer's block and of its bank fills (see
    /// [`HostSyncs`]), each blocking write counted as a sync too.
    pub(crate) syncs: u32,
    /// Blocking host-to-device copies before the launches (the start block of the step).
    pub(crate) host_writes: u32,
}

/// Whether one prefill step was a chunk of rows or the one-row feed of the per-token arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepKind {
    Chunk,
    Feed,
}

/// One prefill step of the report (CPF-7b-C B.8): its first row, its rows, its kind, its host
/// wall time from the chunk open to the status read, and the status word it read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepRecord {
    pub(crate) p0: usize,
    pub(crate) rows: usize,
    pub(crate) kind: StepKind,
    pub(crate) ns: u64,
    pub(crate) status: i32,
}

/// One ExecutorOps::prefill call of the served path (CPF-7b-C B.8, OQ-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PieceRecord {
    pub(crate) tokens: usize,
    pub(crate) ns: u64,
}

/// The plain tag of one chunk-gate verdict (CPF-7b-C B.8): the verdict enum itself is
/// pub(super) in the step, so the report keeps the label only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkVerdictTag {
    Ready,
    TiledLayout,
    Kv8,
    Tiered,
    Q6KWeight,
    NoArena,
}

impl ChunkVerdictTag {
    /// The CPF-8a string of the verdict.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::TiledLayout => "TiledLayout",
            Self::Kv8 => "Kv8",
            Self::Tiered => "Tiered",
            Self::Q6KWeight => "Q6KWeight",
            Self::NoArena => "NoArena",
        }
    }

    /// The tag of the step verdict.
    pub(crate) const fn of(verdict: &super::step::ChunkVerdict) -> Self {
        match verdict {
            super::step::ChunkVerdict::Ready { .. } => Self::Ready,
            super::step::ChunkVerdict::TiledLayout => Self::TiledLayout,
            super::step::ChunkVerdict::Kv8 => Self::Kv8,
            super::step::ChunkVerdict::Tiered => Self::Tiered,
            super::step::ChunkVerdict::Q6KWeight => Self::Q6KWeight,
            super::step::ChunkVerdict::NoArena => Self::NoArena,
        }
    }
}

/// The arm one feed_rows call took (CPF-7b-C B.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathTag {
    Chunk,
    PerToken,
}

impl PathTag {
    /// The CPF-8a string of the arm.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::PerToken => "per_token",
        }
    }
}

/// The bracket list of the job line: [a, b].
fn bracket(items: &[String]) -> String {
    let mut text = String::from("[");
    for (at, item) in items.iter().enumerate() {
        if at > 0 {
            text.push_str(", ");
        }
        text.push_str(item);
    }
    text.push(']');
    text
}

/// The prefill report of one session, accumulated until a reset or an import (CPF-7b-C B.8).
/// Pure host values, so it is its own type: RunReport stays Copy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PrefillReport {
    pub(super) prefix_lookup: PrefixReport,
    steps: Vec<StepRecord>,
    n_prefilled: usize,
    n_fed: usize,
    segmented_parts: usize,
    start_rows: usize,
    /// Whether a chunked call already recorded start_rows in this reset or import epoch.
    epoch_started: bool,
    piece_tokens: Vec<usize>,
    pieces: Vec<PieceRecord>,
    feed_rows_calls: Vec<usize>,
    verdict: Option<ChunkVerdictTag>,
    path_taken: Option<PathTag>,
    m_sess: Option<usize>,
    m_sess_cap: usize,
    arena_bytes: usize,
    part_bytes: usize,
    budget_bytes: usize,
    kv_mode: Option<super::dims::KvMode>,
    layout: crate::family::project::cuda::WeightLayout,
    capture_on: bool,
    logged: bool,
}

impl PrefillReport {
    pub(super) fn observe_prefix_lookup<S, P>(
        &mut self,
        prefix: &PrefixSlot<S, P>,
        prompt: &[u32],
        covered: usize,
    ) {
        self.prefix_lookup = PrefixReport {
            hit: prefix.restored_kind().into(),
            uncached_shared: prefix.uncached_shared(prompt, covered),
            shadow: prefix.shadow_report(),
            // The save-side lifetime totals, sampled at lookup. The resident executor
            // samples at report time; the difference is this job's own saves.
            save_side: prefix.save_side_totals(),
            ..PrefixReport::default()
        };
    }

    /// Records one step: a chunk's rows count as prefilled, a feed's as fed. A step whose status
    /// word was non-zero is still recorded: the record is pushed before the state is revalidated
    /// (CPF-7b-C B.4 3.7).
    pub(crate) fn push_step(&mut self, record: StepRecord) {
        match record.kind {
            StepKind::Chunk => self.n_prefilled += record.rows,
            StepKind::Feed => self.n_fed += record.rows,
        }
        self.steps.push(record);
    }

    pub(crate) fn note_segmented_parts(&mut self, parts: usize) {
        self.segmented_parts = self.segmented_parts.saturating_add(parts);
    }

    pub(crate) fn segmented_parts(&self) -> usize {
        self.segmented_parts
    }

    /// The steps of the session, in order.
    pub(crate) fn steps(&self) -> &[StepRecord] {
        &self.steps
    }

    /// Rows that went through a chunk.
    pub(crate) fn n_prefilled(&self) -> usize {
        self.n_prefilled
    }

    /// Rows that went through the one-row feed.
    /// The arm the last feed_rows call took, once one has run.
    pub(crate) fn path_taken(&self) -> Option<PathTag> {
        self.path_taken
    }

    pub(crate) fn n_fed(&self) -> usize {
        self.n_fed
    }

    /// Committed rows at the first chunked call of the current reset or import epoch.
    pub(crate) fn start_rows(&self) -> usize {
        self.start_rows
    }

    /// Sets start_rows, called by the driver at the first chunked call of an epoch.
    pub(crate) fn set_start_rows(&mut self, rows: usize) {
        self.start_rows = rows;
    }

    /// Records the committed rows at the FIRST chunked call of a reset or import epoch and
    /// reports whether this call was that one.
    pub(crate) fn note_chunked_epoch_start(&mut self, rows: usize) -> bool {
        if self.epoch_started {
            return false;
        }
        self.epoch_started = true;
        self.start_rows = rows;
        true
    }

    /// The tokens of each ExecutorOps::prefill call.
    pub(crate) fn piece_tokens(&self) -> &[usize] {
        &self.piece_tokens
    }

    /// One record per ExecutorOps::prefill call.
    pub(crate) fn pieces(&self) -> &[PieceRecord] {
        &self.pieces
    }

    /// Records one ExecutorOps::prefill call: its tokens and its host wall time.
    pub(crate) fn record_piece(&mut self, tokens: usize, ns: u64) {
        self.piece_tokens.push(tokens);
        self.pieces.push(PieceRecord { tokens, ns });
    }

    /// Records one feed_rows call and the verdict it routed on (CPF-7b-C B.8).
    pub(crate) fn note_feed_rows(&mut self, verdict: &super::step::ChunkVerdict, rows: usize) {
        self.verdict = Some(ChunkVerdictTag::of(verdict));
        self.path_taken = Some(match verdict {
            super::step::ChunkVerdict::Ready { .. } => PathTag::Chunk,
            _ => PathTag::PerToken,
        });
        self.m_sess = match verdict {
            super::step::ChunkVerdict::Ready { m_sess } => Some(*m_sess),
            _ => None,
        };
        self.feed_rows_calls.push(rows);
    }

    /// The rows of each feed_rows call.
    pub(crate) fn feed_rows_calls(&self) -> &[usize] {
        &self.feed_rows_calls
    }

    /// Empties the session and refills the static fields, for a reset or an import: the report
    /// of the previous epoch must not cross into the next one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refill(
        &mut self,
        m_sess_cap: usize,
        arena_bytes: usize,
        part_bytes: usize,
        budget_bytes: usize,
        kv_mode: super::dims::KvMode,
        layout: crate::family::project::cuda::WeightLayout,
        capture_on: bool,
    ) {
        self.clear();
        self.verdict = None;
        self.path_taken = None;
        self.m_sess = None;
        self.feed_rows_calls.clear();
        self.m_sess_cap = m_sess_cap;
        self.arena_bytes = arena_bytes;
        self.part_bytes = part_bytes;
        self.budget_bytes = budget_bytes;
        self.kv_mode = Some(kv_mode);
        self.layout = layout;
        self.capture_on = capture_on;
        self.logged = false;
    }

    /// Takes the once-per-job line: Some once, then None until the next refill.
    pub(crate) fn take_job_line(&mut self) -> Option<String> {
        if self.logged {
            return None;
        }
        self.logged = true;
        Some(self.job_line())
    }

    /// The one engine log line of a prefill job (CPF-7b-C B.8, OQ-13).
    pub(crate) fn job_line(&self) -> String {
        let m_sess = match self.m_sess {
            Some(rows) => rows.to_string(),
            None => "none".to_owned(),
        };
        let path = self.path_taken.map_or("none", PathTag::label);
        let verdict = self.verdict.map_or("none", ChunkVerdictTag::label);
        let steps: Vec<String> = self
            .steps
            .iter()
            .map(|step| match step.kind {
                StepKind::Chunk => format!("Chunk({}, {})", step.p0, step.rows),
                StepKind::Feed => format!("Feed({})", step.p0),
            })
            .collect();
        let piece_tokens: Vec<String> = self.piece_tokens.iter().map(usize::to_string).collect();
        let calls: Vec<String> = self.feed_rows_calls.iter().map(usize::to_string).collect();
        let piece_ns: Vec<String> = self
            .pieces
            .iter()
            .map(|piece| piece.ns.to_string())
            .collect();
        let chunk_ns: Vec<String> = self
            .steps
            .iter()
            .filter(|step| step.kind == StepKind::Chunk)
            .map(|step| step.ns.to_string())
            .collect();
        let mut line = format!(
            "q36 cuda: prefill job m_sess {m_sess} m_sess_cap {cap} path {path} verdict {verdict} pieces {pieces} piece_tokens {tokens} calls {calls} steps {steps} rows {prefilled}+{fed} piece_ns {pns} chunk_ns {cns}",
            cap = self.m_sess_cap,
            pieces = self.pieces.len(),
            tokens = bracket(&piece_tokens),
            calls = bracket(&calls),
            steps = bracket(&steps),
            prefilled = self.n_prefilled,
            fed = self.n_fed,
            pns = bracket(&piece_ns),
            cns = bracket(&chunk_ns),
        );
        if self.segmented_parts() != 0 {
            line.push_str(&format!(" segmented_parts {}", self.segmented_parts()));
        }
        line
    }

    /// The static fields of the job line, for the window rows.
    pub(crate) fn statics(&self) -> (usize, usize, usize, usize, bool) {
        (
            self.m_sess_cap,
            self.arena_bytes,
            self.part_bytes,
            self.budget_bytes,
            self.capture_on,
        )
    }

    /// Empties every list and count. start_rows is kept: the driver sets it at the first chunked
    /// call of the next epoch. The serve surface static fields are refilled by its own reset and
    /// import hooks (C4b).
    pub(crate) fn clear(&mut self) {
        self.prefix_lookup = PrefixReport::default();
        self.steps.clear();
        self.epoch_started = false;
        self.n_prefilled = 0;
        self.n_fed = 0;
        self.segmented_parts = 0;
        self.piece_tokens.clear();
        self.pieces.clear();
    }
}
