//! The worker-driven resume round: the real save-point plan, the real store and the chunk
//! driver (the serve's prefill path) on fakes, through a three-turn conversation, a mid-turn
//! suffix, an unrelated request and a return to the conversation — the sequence the review's
//! S1 (a wedged budget stops reuse advancing) and S2 (the block lost to a Tip-restoring turn)
//! live in. Everything is observed through the worker's own job usage: `reused_tokens` per
//! request, the way the serve's request log sees it.

use std::time::{Duration, Instant};

use super::{CudaRules, FlashCudaExecutor};
use crate::family::flash_next::cuda::driver::DriverShape;
use crate::family::flash_next::cuda::driver::ResumesState;
use crate::family::flash_next::cuda::driver::StateSlot;
use crate::family::flash_next::cuda::driver::chunk::ChunkDriver;
use crate::family::flash_next::cuda::driver::fakes::{
    FakeChunkDevice, FakeFeeds, SAMPLER_VOCAB, shape,
};
use crate::family::flash_next::cuda::resume;
use crate::sched::job::JobEvent;
use crate::sched::worker::spawn_engine;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// The three-token tokenizer of the worker tests; the ids, not the text, are what is checked.
const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

/// The conversation of the round, token ids inside the fake's six-id sampler vocabulary.
fn turn(ids: &[u32]) -> Vec<u32> {
    ids.to_vec()
}

/// Submits one prompt with its boundary and returns the job's usage: the reused tokens the
/// worker's restore covered, after the completion finished.
fn request(engine: &crate::sched::worker::Engine, prompt: &[u32], boundary: usize) -> (u32, u32) {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
    let generation =
        crate::family::executor::GenerationState::new(IncrementalDecoder::new(&tokenizer))
            .with_max_tokens(1);
    let (job, mut handle) = crate::sched::job::Job::new(prompt.to_vec(), generation).expect("job");
    let job = job.with_prefix_boundary(boundary).expect("boundary");
    engine.handle().try_submit(job).expect("submit");
    let deadline = Instant::now() + Duration::from_secs(30);
    let usage = loop {
        assert!(Instant::now() < deadline, "timed out waiting for the job");
        match handle.events.try_recv() {
            Ok(JobEvent::Started) | Ok(JobEvent::Batch(_)) => {}
            Ok(JobEvent::Finished(usage)) => break usage,
            Ok(JobEvent::Error(error)) => panic!("the job failed: {error:?}"),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(5))
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                panic!("the engine dropped the job channel")
            }
        }
    };
    (usage.reused_tokens, usage.completion_tokens)
}

/// The three-turn conversation round, on a prefill piece of 4 so a turn that restores a
/// Tip still rolls a Block (with 16-row pieces the moved cut lands at or above the tip and
/// the worker skips it). The suffix grows PAST the old tip and re-words the seam, so the
/// restoring turn's own saves cannot land on the stale Tip's key and heal it: only the
/// drop rule can retire the stale entry. Mutations this test fails under (verified by
/// editing the source and rerunning, 2026-11-12):
/// (i) `with_recent_longer_protection` removed — turn 3 (the suffix) restores 0: the Tip
///     save of turn 2 evicted the block (the review's S2).
/// (ii) the pre-fix budget charge restored (pre-room, stale entries counted, no path
///     reclamation) — turn 3 restores 23, turn 1's tip, because turn 2's deeper Tip save was
///     denied: reuse stops advancing (the review's S1 wedge).
/// (iii) the restore-time `drop_above` call removed — turn 4 restores the stale Tip of the
///     suffixed turn 3 (29 tokens of rows the suffix overwrote) instead of the rolled
///     Block: reuse advances through an entry whose rows no longer hold its tokens.
#[test]
fn the_worker_drives_three_turns_a_suffix_and_a_switch() {
    resume::arm_for_tests();
    let engine = spawn_engine(4, |_| {
        // A fake device with real fixed bytes (1 KiB), so a snapshot costs host RAM and the
        // budget's ratio bites the way the serve's 256 MiB line does against its entries.
        let device = FakeChunkDevice {
            state: vec![(StateSlot::DeltaRec(0), vec![7; 1024])],
            ..Default::default()
        };
        let driver = ChunkDriver::new(
            device,
            FakeFeeds::default(),
            DriverShape {
                positions: 160,
                ..shape()
            },
        )
        .expect("chunk driver");
        // The serve's prefill piece and the resume store at a scale the fake device holds:
        // the block lattice shrinks 64 -> 8 and the budget keeps the serve's ratio (the
        // 256 MiB line against 110.8 MiB entries — one survivor plus the new entry fits,
        // two survivors plus one denies).
        let entry_bytes = ResumesState::resume_host_bytes(&driver);
        let (executor, session) = FlashCudaExecutor::with_host_budget_for_tests(
            driver,
            CudaRules {
                eos: std::sync::Arc::from([]),
                sampler_vocab: SAMPLER_VOCAB,
            },
            2 * entry_bytes,
        )
        .expect("executor");
        Ok((executor.with_block_spacing_for_tests(8), session))
    })
    .expect("engine");

    // The conversation: A, A extended (turn 2), the extension re-worded at its end (the
    // mid-turn suffix), an unrelated request, the conversation again, and one more turn of
    // it — six requests whose reused tokens are the whole story.
    let a = turn(&[
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    ]);
    let mut b = a.clone();
    b.extend_from_slice(&[3, 4, 3, 4, 3, 4]); // turn 2 grows the conversation
    // The suffix re-renders the seam AND grows past the old tip: its prompt diverges from
    // the tip key inside the boundary, and its own new tip (32) lands past the stale tip's
    // key (29), so nothing re-saves the stale key.
    let mut c = b.clone();
    c.truncate(28);
    c.extend_from_slice(&[5, 5, 5, 5]);
    let d = turn(&[
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ]);
    let e = b.clone(); // the conversation's second turn again, after the suffix
    let mut f = e.clone();
    f.extend_from_slice(&[1, 1]);

    let boundary = |p: &[u32]| p.len() - 1;
    let (reused, _) = request(&engine, &a, boundary(&a));
    assert_eq!(reused, 0, "turn 1 serves cold");
    let (reused, _) = request(&engine, &b, boundary(&b));
    assert_eq!(reused, 23, "turn 2 restores the tip of turn 1");
    let (reused, _) = request(&engine, &c, boundary(&c));
    assert_eq!(
        reused, 27,
        "the suffix restores the rolled Block of turn 2 (not the re-worded tip): turn 2's \
        Tip save kept the Block beside it"
    );
    let (reused, _) = request(&engine, &e, boundary(&e));
    assert_eq!(
        reused, 27,
        "the conversation's second turn after the suffix restores the rolled Block, not the \
        stale Tip the suffix overwrote (the drop rule)"
    );
    let (reused, _) = request(&engine, &f, boundary(&f));
    assert_eq!(
        reused, 29,
        "reuse advances: turn 5 restores turn 4's saved tip"
    );
    let (reused, completion) = request(&engine, &d, boundary(&d));
    assert_eq!(
        reused, 0,
        "the unrelated request serves cold after the switch"
    );
    assert_eq!(completion, 1, "every job completed one token");

    let report = engine.shutdown();
    let _ = report;
}
