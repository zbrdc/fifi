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
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::{EVENT_CAPACITY, Job, JobEvent, JobUsage};
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
        })
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
