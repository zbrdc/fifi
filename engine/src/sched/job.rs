//! One generation job: the prompt, the generation state, and the bounded output channel.
//!
//! Design: the scheduler design ("Threads and channels"). A job carries portable data only.
//!
//! Terminal events: the output channel holds `EVENT_CAPACITY` events. The engine sends a data
//! event (a batch) only while at least two slots are free, so the last slot always remains for
//! the one terminal event of the job (`Finished` or `Error`). A terminal event therefore never
//! lacks room. A job whose batch does not fit is cancelled as a slow client and gets no
//! terminal event; its channel closes with the cancel flag set.
//!
//! Disconnect: dropping the `JobHandle` cancels the job. A queued job never starts. A running
//! job stops after its current pass.

use crate::EngError;
use crate::family::executor::GenerationState;
use crate::family::session::SaveKind;
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

/// Capacity of the bounded queue in front of one engine (scheduler.md, item 4).
pub(crate) const QUEUE_CAPACITY: usize = 32;

/// Capacity of the bounded output channel of one job (scheduler.md, item 3).
pub(crate) const EVENT_CAPACITY: usize = 1_024;

/// Token counts of a finished job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JobUsage {
    /// Tokens in the submitted prompt.
    pub(crate) prompt_tokens: u32,
    /// Tokens in the batches that the executor committed, including a final EOS token.
    pub(crate) completion_tokens: u32,
    /// Prompt tokens served from the saved prefix state (decision 0036, addendum 1). It is 0
    /// unless the job carries a prefix boundary and the executor restored a saved prefix.
    pub(crate) reused_tokens: u32,
    /// The prefix-store report of the job (decision 0036, addendum 2; packet SERVE-2-c item
    /// 3), taken after the prefill: the store of a job with a prefix boundary, or `None`
    /// when the executor keeps no prefix report (the default of `ExecutorOps::prefix_report`).
    pub(crate) prefix: Option<PrefixReport>,
}

/// The kind of the stored prefix entry that a job restored (decision 0036, addendum 2; packet
/// SERVE-2-c item 3), for the request-log field `prefix_hit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixHit {
    /// No stored entry matched the prompt.
    Miss,
    /// The job restored a pinned root.
    Root,
    /// The job restored a periodic checkpoint.
    Periodic,
    /// The job restored a tip entry.
    Tip,
    /// The job restored a block cut (KV-v1 slice 2): the rolling intermediate entry of the
    /// lineage, so a rewritten prompt's reuse is distinguishable from a tip's exact extension.
    Block,
}

impl PrefixHit {
    /// The name of the request-log field `prefix_hit`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            PrefixHit::Miss => "miss",
            PrefixHit::Root => "root",
            PrefixHit::Periodic => "periodic",
            PrefixHit::Tip => "tip",
            PrefixHit::Block => "block",
        }
    }
}

/// The prefix-hit class of the entry a restore took (fix L5): `Some(kind)` maps to its own
/// class and `None` (nothing restored) to `Miss`. The one mapping that the executors share.
impl From<Option<SaveKind>> for PrefixHit {
    fn from(kind: Option<SaveKind>) -> Self {
        match kind {
            Some(SaveKind::Root) => PrefixHit::Root,
            Some(SaveKind::Periodic) => PrefixHit::Periodic,
            Some(SaveKind::Tip) => PrefixHit::Tip,
            Some(SaveKind::Block) => PrefixHit::Block,
            None => PrefixHit::Miss,
        }
    }
}

/// The prefix-store facts of one job (decision 0036, addendum 2; packet SERVE-2-c item 3):
/// what the engine's store holds after the job's save points and what the job restored. The
/// worker takes it after the prefill, before the first decode, and carries it in `JobUsage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixReport {
    /// The entries the store holds after the job's save points, pinned roots and LRU entries
    /// together.
    pub(crate) entries: u32,
    /// The charged bytes of the stored snapshots (the measured cost, report M0): each entry
    /// charges the KV bytes of its positions plus its fixed state bytes, and the executor sums
    /// the charge of its own entries. The store never sizes a snapshot, and an executor
    /// without a charge reports 0.
    pub(crate) bytes: u64,
    /// The kind of the entry the job restored, or `Miss` when nothing restored.
    pub(crate) hit: PrefixHit,
    /// Prompt tokens shared with the longest stored key beyond the restored prefix: tokens
    /// that were shared with a stored key but not covered by a checkpoint. It is 0 when the
    /// longest stored key equals the restored prefix.
    pub(crate) uncached_shared: u32,
    /// The shadow block-LCP observation of the job's lookup (KV-v1 slice 1), or `None` when
    /// the shadow stayed off. It rides beside the restore facts: what a block-granular
    /// restore could have used, for the request log.
    pub(crate) shadow: Option<crate::family::session::ShadowLcp>,
    /// SF1 attribution: candidates the store dropped after an adoption `MemoryPlan`
    /// denial, and entries spent by the radix-order eviction. Zero on executors without
    /// a store - and the CUDA path reports 0 by design: its store still runs the SF1
    /// retry, but it does not surface the counters into its report. LIFETIME store totals, never
    /// reset. Each request line carries ONE sample, taken right after the request's own (last)
    /// restore on the batch worker and after its prefill on the single worker; lines are written
    /// at completion, so attribute a request's own restore drops against the sample of the
    /// restore that ran just before it (`restore_drops` rises only at restores, so ordering
    /// samples by value recovers restore order) - never against the previous line. `evictions`
    /// counts only `evict_lru_one` spends and also rises between samples through other slots'
    /// denial evictions; a self-spend counts BOTH counters once, so drops + evictions over-counts
    /// entries lost by the number of self-evictions.
    /// A rise in `restore_drops` between a request's restore sample and the previous
    /// RESTORE's sample points at the adoption denial, not at a store that never held
    /// the anchor.
    pub(crate) restore_drops: u32,
    pub(crate) evictions: u32,
    /// B2 attribution: the store's save-side lifetime totals (room churn, refusals,
    /// duplicate uses, block rolls, in-place replaces). Zero on executors without a
    /// store. The same one-sample-per-request-line attribution rule as the SF1 pair;
    /// `save_side.room_removals` is the churn `evictions` cannot see - a store can lose
    /// every mark of a lineage to other saves' room pressure while reporting
    /// `evictions` 0.
    pub(crate) save_side: crate::family::session::SaveSideTotals,
}

impl Default for PrefixReport {
    /// The report of an executor that keeps no prefix state: all zero with a `Miss`.
    fn default() -> Self {
        Self {
            entries: 0,
            bytes: 0,
            hit: PrefixHit::Miss,
            uncached_shared: 0,
            shadow: None,
            restore_drops: 0,
            evictions: 0,
            save_side: crate::family::session::SaveSideTotals::default(),
        }
    }
}

/// One message from the engine thread to the owner of a job.
#[derive(Debug, PartialEq)]
pub(crate) enum JobEvent {
    /// One committed batch. The batch with `finish` set is the last one.
    Batch(DecodeBatch),
    /// The usage of the job. It follows the last batch and always has room in the channel.
    Finished(JobUsage),
    /// The job failed. No batch or usage follows. It always has room in the channel.
    Error(EngError),
}

/// The engine side of a job. It moves to the engine thread.
pub(crate) struct Job {
    pub(super) prompt: Vec<u32>,
    pub(super) generation: GenerationState,
    pub(super) events: mpsc::Sender<JobEvent>,
    /// The number of leading prompt tokens that end at the last message boundary. `None` turns
    /// prefix reuse off for this job.
    pub(super) prefix_boundary: Option<usize>,
    /// The number of leading prompt tokens that end at the first system message, when the job
    /// carries one. It is set only beside a prefix boundary, and it lies inside it.
    pub(super) root_boundary: Option<usize>,
}

/// The owner side of a job: its event receiver and its cancellation handle.
/// Dropping it cancels the job.
pub(crate) struct JobHandle {
    /// The bounded output channel. Only this job's events arrive here.
    pub(crate) events: mpsc::Receiver<JobEvent>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    /// Creates a job and its handle. The prompt must hold at least one token.
    pub(crate) fn new(
        prompt: Vec<u32>,
        generation: GenerationState,
    ) -> Result<(Job, JobHandle), EngError> {
        if prompt.is_empty() {
            return Err(EngError::BadRequest("prompt has no tokens".to_string()));
        }
        let (events, receiver) = mpsc::channel(EVENT_CAPACITY);
        let cancel = Arc::clone(&generation.cancel);
        Ok((
            Job {
                prompt,
                generation,
                events,
                prefix_boundary: None,
                root_boundary: None,
            },
            JobHandle {
                events: receiver,
                cancel,
            },
        ))
    }

    /// Marks the first `boundary` prompt tokens as the reusable prefix. The engine restores a
    /// saved prefix that matches them and saves the state after them. The boundary must leave at
    /// least one token after it. A job that already carries a root keeps the rule `root <
    /// boundary`: a new boundary at or before that root is refused.
    pub(crate) fn with_prefix_boundary(mut self, boundary: usize) -> Result<Job, EngError> {
        if boundary == 0 || boundary >= self.prompt.len() {
            return Err(EngError::BadRequest(
                "prefix boundary must lie inside the prompt".to_string(),
            ));
        }
        if self.root_boundary.is_some_and(|root| root >= boundary) {
            return Err(EngError::BadRequest(
                "the prefix boundary must lie after the root boundary".to_string(),
            ));
        }
        self.prefix_boundary = Some(boundary);
        Ok(self)
    }

    /// Marks the first `root` prompt tokens as the root save point of decision 0036, addendum
    /// 2: the engine saves them as a pinned root after prefilling them. A root needs a prefix
    /// boundary first (`with_prefix_boundary`) and must lie inside it: `0 < root <
    /// prefix_boundary`. The 2,048-token minimum of the decision is the worker's rule, not the
    /// job's.
    pub(crate) fn with_root_boundary(mut self, root: usize) -> Result<Job, EngError> {
        let Some(boundary) = self.prefix_boundary else {
            return Err(EngError::BadRequest(
                "a root boundary needs a prefix boundary first".to_string(),
            ));
        };
        if root == 0 || root >= boundary {
            return Err(EngError::BadRequest(
                "the root boundary must lie inside the reusable prefix".to_string(),
            ));
        }
        self.root_boundary = Some(root);
        Ok(self)
    }
}

impl JobHandle {
    /// Asks the engine to stop this job. A queued job does not start. A running job stops after its current pass.
    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// True after `cancel` or after the engine cancelled the job.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Sends a data event. It keeps the last slot free for the terminal event.
/// Returns false when the channel is full (slow client) or closed.
pub(super) fn send_data(events: &mpsc::Sender<JobEvent>, event: JobEvent) -> bool {
    events.capacity() >= 2 && events.try_send(event).is_ok()
}

/// Sends the one terminal event. The slot that `send_data` keeps free holds it.
pub(super) fn send_terminal(events: &mpsc::Sender<JobEvent>, event: JobEvent) {
    let _ = events.try_send(event);
}

/// Sends the events of a job that ended before any generation: a `Cancelled` batch and a usage
/// event with no completion tokens.
pub(super) fn send_cancelled(events: &mpsc::Sender<JobEvent>, prompt_tokens: u32) {
    send_data(
        events,
        JobEvent::Batch(DecodeBatch {
            tokens: Vec::new(),
            text: String::new(),
            finish: Some(DecodeFinish::Cancelled),
        }),
    );
    send_terminal(
        events,
        JobEvent::Finished(JobUsage {
            prompt_tokens,
            completion_tokens: 0,
            reused_tokens: 0,
            prefix: None,
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::{EVENT_CAPACITY, Job, JobEvent, JobUsage, PrefixHit, PrefixReport};
    use crate::EngError;
    use crate::family::executor::GenerationState;
    use crate::tokenizer::{IncrementalDecoder, Tokenizer};
    use std::sync::atomic::Ordering;

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

    fn generation() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    fn usage_event() -> JobEvent {
        JobEvent::Finished(JobUsage {
            prompt_tokens: 1,
            completion_tokens: 0,
            reused_tokens: 0,
            prefix: None,
        })
    }

    /// The default report of an executor that keeps no prefix state is all zero with a `Miss`,
    /// and each hit kind carries its request-log name (packet SERVE-2-c item 3).
    #[test]
    fn a_default_prefix_report_is_all_zero_with_a_miss_and_the_hits_have_names() {
        assert_eq!(
            PrefixReport::default(),
            PrefixReport {
                entries: 0,
                bytes: 0,
                hit: PrefixHit::Miss,
                uncached_shared: 0,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default()
            },
            "a literal, not the default under test"
        );
        let names = [
            (PrefixHit::Miss, "miss"),
            (PrefixHit::Root, "root"),
            (PrefixHit::Periodic, "periodic"),
            (PrefixHit::Tip, "tip"),
        ];
        for (hit, name) in names {
            assert_eq!(hit.as_str(), name, "{hit:?}");
        }
    }

    /// The `From` mapping of the restored kind (fix L5): every `SaveKind` names its own class
    /// and `None` names a `Miss`. The `Periodic` row is the one no other test reaches, because
    /// the store-level tests restore roots and tips only.
    ///
    /// Mutation check: mapping `Some(SaveKind::Periodic)` to `PrefixHit::Tip` fails the
    /// periodic row.
    #[test]
    fn the_restored_kind_maps_to_its_prefix_hit_class() {
        use crate::family::session::SaveKind;
        let hits = [
            (None, PrefixHit::Miss),
            (Some(SaveKind::Root), PrefixHit::Root),
            (Some(SaveKind::Periodic), PrefixHit::Periodic),
            (Some(SaveKind::Tip), PrefixHit::Tip),
        ];
        for (kind, want) in hits {
            assert_eq!(
                PrefixHit::from(kind),
                want,
                "kind {kind:?} must name its own class"
            );
        }
    }

    #[test]
    fn a_job_rejects_an_empty_prompt() {
        let error = Job::new(Vec::new(), generation())
            .err()
            .expect("an empty prompt must fail");
        assert!(matches!(error, EngError::BadRequest(_)), "got {error:?}");
    }

    #[test]
    fn a_prefix_boundary_must_leave_a_token_after_it_and_cover_one_before_it() {
        let cases: [(usize, bool); 5] = [(0, false), (1, true), (2, true), (3, false), (4, false)];
        for (boundary, accepted) in cases {
            let (job, _handle) = Job::new(vec![5, 6, 7], generation()).expect("job");
            let result = job.with_prefix_boundary(boundary);
            assert_eq!(result.is_ok(), accepted, "boundary {boundary}");
            if let Ok(job) = result {
                assert_eq!(job.prefix_boundary, Some(boundary), "stored boundary");
            }
        }
    }

    #[test]
    fn a_new_job_carries_no_prefix_boundary() {
        let (job, _handle) = Job::new(vec![5, 6, 7], generation()).expect("job");
        assert_eq!(job.prefix_boundary, None);
        assert_eq!(job.root_boundary, None, "a fresh job carries no root");
    }

    #[test]
    fn a_root_boundary_needs_a_prefix_boundary_first() {
        let (job, _handle) = Job::new(vec![5, 6, 7], generation()).expect("job");
        let error = job
            .with_root_boundary(1)
            .err()
            .expect("a root without a tip must fail");
        assert!(matches!(error, EngError::BadRequest(_)), "got {error:?}");
    }

    #[test]
    fn a_new_prefix_boundary_must_stay_after_the_root_boundary() {
        // The tip 4 and the root 3 are valid. A new boundary of 3 or 2 would leave the root at
        // or after the tip, so the job refuses it. A boundary of 5 keeps the rule.
        let cases: [(usize, bool); 3] = [(2, false), (3, false), (5, true)];
        for (boundary, accepted) in cases {
            let (job, _handle) = Job::new(vec![5, 6, 7, 8, 9, 10], generation()).expect("job");
            let job = job.with_prefix_boundary(4).expect("tip");
            let job = job.with_root_boundary(3).expect("root");
            let result = job.with_prefix_boundary(boundary);
            assert_eq!(result.is_ok(), accepted, "boundary {boundary}");
            if let Ok(job) = result {
                assert_eq!(job.prefix_boundary, Some(boundary), "stored tip");
                assert_eq!(job.root_boundary, Some(3), "the root stays");
            }
        }
    }

    #[test]
    fn a_root_boundary_must_lie_inside_the_reusable_prefix() {
        // The tip boundary is 3, so a root of 1 or 2 lies inside it.
        let cases: [(usize, bool); 5] = [(0, false), (1, true), (2, true), (3, false), (4, false)];
        for (root, accepted) in cases {
            let (job, _handle) = Job::new(vec![5, 6, 7, 8], generation()).expect("job");
            let job = job.with_prefix_boundary(3).expect("boundary");
            let result = job.with_root_boundary(root);
            assert_eq!(result.is_ok(), accepted, "root {root}");
            if let Ok(job) = result {
                assert_eq!(job.root_boundary, Some(root), "stored root");
            }
        }
    }

    #[test]
    fn the_job_channel_holds_1024_events_and_then_reports_full() {
        let (job, _handle) = Job::new(vec![0], generation()).expect("job");
        for index in 0..EVENT_CAPACITY {
            job.events
                .try_send(usage_event())
                .unwrap_or_else(|error| panic!("event {index} did not fit: {error:?}"));
        }
        assert_eq!(EVENT_CAPACITY, 1_024);
        assert!(
            job.events.try_send(usage_event()).is_err(),
            "event 1025 fit into the channel"
        );
    }

    #[test]
    fn dropping_the_handle_cancels_the_job() {
        let (job, handle) = Job::new(vec![0], generation()).expect("job");
        assert!(!job.generation.cancel.load(Ordering::SeqCst));
        drop(handle);
        assert!(
            job.generation.cancel.load(Ordering::SeqCst),
            "a dropped handle left the job running"
        );
    }

    #[test]
    fn the_handle_cancels_the_flag_that_the_executor_reads() {
        let (job, handle) = Job::new(vec![0], generation()).expect("job");
        assert!(!handle.is_cancelled(), "a new job starts cancelled");
        handle.cancel();
        assert!(handle.is_cancelled(), "the handle lost its own cancel");
        assert!(
            job.generation.cancel.load(Ordering::SeqCst),
            "the generation state does not see the cancel"
        );
    }
}
