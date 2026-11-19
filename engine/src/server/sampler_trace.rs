//! The per-request sampler-trace ring, its wire packet, and the read route.
//!
//! Contract: capture is opt-in twice — the process setting `sample_trace`
//! permits it, and a request arms it with `x-sample-trace: true` before
//! generation begins; nothing is recorded retroactively. Buffers register in
//! the ring at submit with their reply's metadata, so in-progress captures
//! count against the global budget. The ring keeps the last
//! [`MAX_RETAINED_REPLIES`] replies under [`GLOBAL_POSITION_BUDGET`] positions
//! (drop-oldest; a flooded ring may drop a running request's trace, and the
//! route then reports not retained).
//!
//! Packet v2 carries, per position, the token id, rank, kept count,
//! greedy flag, probabilities, and the reply's own generated piece text
//! with its raw-text byte offset — reasoning and tool-call text included,
//! since that is what the model emitted. The prompt is never copied.
//! Authorization is the process API key,
//! exactly like every other `/v1` route: one key serves the whole process, so
//! per-user trace isolation is NOT provided — any key holder can read any
//! reply's trace by id. The position budgets bound memory by the serialized
//! worst case: a test-verified byte bound per position times the position
//! counts.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::error::ServerError;
use super::state::AppState;
use crate::family::sampler::{CaptureBuffer, CaptureHandle};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// The per-request absolute position budget. 8192 positions at the serialized
/// worst case stay under 3.2 MiB.
pub(crate) const PER_REQUEST_POSITION_LIMIT: usize = 8_192;

/// How many replies the ring retains. The GUI's details view reads one reply's
/// trace right after its stream ends, so the window is small on purpose.
const MAX_RETAINED_REPLIES: usize = 4;

/// The global position budget across the ring, in-progress captures included:
/// [`MAX_RETAINED_REPLIES`] full requests.
const GLOBAL_POSITION_BUDGET: usize = MAX_RETAINED_REPLIES * PER_REQUEST_POSITION_LIMIT;

/// What the packet needs about a reply besides its positions. Registered at
/// submit, when the reply head and the resolved settings are at hand.
#[derive(Clone)]
pub(crate) struct TraceMeta {
    pub(crate) model: String,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) top_k: u32,
}

struct TraceEntry {
    id: String,
    meta: TraceMeta,
    buffer: CaptureHandle,
}

/// Replays the decode batches of one traced reply through an incremental
/// decoder over the same tokenizer the engine used, attaching each token's
/// piece and its byte offset in the raw generated text to the sampler's
/// recorded position. The GUI maps raw offsets into the reply's fields by
/// replaying the piece stream it already receives; the packet never guesses
/// that mapping. Alignment is write-only telemetry: a disagreement latches
/// `alignment_error` on the buffer and stops the feed — the reply itself is
/// unaffected.
pub(crate) struct TraceAligner {
    buffer: CaptureHandle,
    decoder: IncrementalDecoder,
    /// Set once the feed stopped; further batches are ignored.
    stopped: bool,
}

impl TraceAligner {
    pub(crate) fn new(tokenizer: &Tokenizer, buffer: CaptureHandle) -> Self {
        Self {
            buffer,
            decoder: IncrementalDecoder::new(tokenizer),
            stopped: false,
        }
    }

    /// Feeds one batch's tokens in order. The first decode failure or token
    /// the sampler never recorded latches `alignment_error` and stops the
    /// feed: offsets already written stay valid, later positions stay bare.
    pub(crate) fn on_batch(&mut self, tokens: &[u32]) {
        if self.stopped {
            return;
        }
        for &id in tokens {
            let piece = match self.decoder.push(id) {
                Ok(piece) => piece,
                Err(_) => {
                    // A decode failure in the replay is an alignment stop
                    // by contract: the latch must show in the packet, not
                    // just in the missing offsets.
                    self.latch_alignment_error();
                    return self.stop();
                }
            };
            let aligned = {
                let mut buffer = self
                    .buffer
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                buffer.annotate(id, &piece)
            };
            if !aligned {
                return self.stop();
            }
        }
    }

    /// Flushes the decoder's pending text into the trailer after the last
    /// batch. It belongs to no single position — typically a partial UTF-8
    /// sequence the output limit cut mid-character.
    pub(crate) fn finish(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        match self.decoder.finish() {
            Ok(text) => {
                self.buffer
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_trailer(&text);
            }
            // A tail that cannot decode is the same contract break as a
            // mid-stream failure: latched, with the written offsets kept.
            Err(_) => self.latch_alignment_error(),
        }
    }

    fn latch_alignment_error(&self) {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        buffer.alignment_error = true;
    }

    fn stop(&mut self) {
        self.stopped = true;
    }
}

/// The drop-oldest ring of reply id to capture buffer and metadata.
#[derive(Default)]
pub(crate) struct SamplerTraceRing {
    entries: Mutex<VecDeque<TraceEntry>>,
}

impl SamplerTraceRing {
    /// Registers a capture buffer under its reply id, replacing a same-id
    /// registration, then evicts oldest entries until both budgets hold.
    pub(crate) fn register(&self, id: &str, meta: TraceMeta, buffer: CaptureHandle) {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.retain(|entry| entry.id != id);
        entries.push_back(TraceEntry {
            id: id.to_owned(),
            meta,
            buffer,
        });
        while entries.len() > MAX_RETAINED_REPLIES
            || total_positions(&entries) > GLOBAL_POSITION_BUDGET
        {
            if entries.pop_front().is_none() {
                break;
            }
        }
    }

    /// The capture buffer and metadata of one reply id, when still retained.
    pub(crate) fn get(&self, id: &str) -> Option<(TraceMeta, CaptureHandle)> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| (entry.meta.clone(), Arc::clone(&entry.buffer)))
    }
}

fn total_positions(entries: &VecDeque<TraceEntry>) -> usize {
    entries
        .iter()
        .map(|entry| {
            entry
                .buffer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .positions
                .len()
        })
        .sum()
}

#[derive(Deserialize)]
pub(crate) struct TraceQuery {
    id: Option<String>,
}

/// The trace packet of one reply: version 1, ids and probabilities only.
/// Piece alignment (offset-based text mapping) is a later packet revision;
/// this version stays joinable on `token_id`.
fn packet(reply_id: &str, meta: &TraceMeta, buffer: &CaptureBuffer) -> Value {
    let positions: Vec<Value> = buffer
        .positions
        .iter()
        .map(|recorded| {
            json!({
                "token_id": recorded.token,
                "rank": recorded.rank,
                "kept": recorded.kept,
                "greedy": recorded.greedy,
                "selected_prob": recorded.selected_prob,
                "candidates": recorded
                    .candidates
                    .iter()
                    .map(|&(id, prob)| json!({ "token_id": id, "prob": prob }))
                    .collect::<Vec<_>>(),
                // The alignment pair: null until the replay reaches
                // this position, or forever when alignment stopped early.
                "offset": recorded.offset,
                "piece": recorded.piece,
            })
        })
        .collect();
    let aligned = buffer
        .positions
        .iter()
        .filter(|recorded| recorded.offset.is_some())
        .count();
    json!({
        "packet": 2,
        "reply_id": reply_id,
        "model": meta.model,
        "sampling": {
            "temperature": meta.temperature,
            "top_p": meta.top_p,
            "top_k": meta.top_k,
        },
        "capture_scope": "decode",
        "truncated": buffer.truncated,
        "captured_positions": buffer.positions.len(),
        // Alignment: offsets are bytes into the raw generated text
        // (the concatenation of pieces); the GUI maps raw offsets into the
        // reply's fields by replaying the piece stream it receives.
        "raw_byte_len": buffer.raw_len,
        "aligned_positions": aligned,
        "alignment_error": buffer.alignment_error,
        "trailer": buffer.trailer,
        "positions": positions,
    })
}

/// `GET /v1/sampler-trace?id=<reply id>` — one reply's packet, no-store.
/// Unknown or evicted ids report not found, so a dropped trace is explicit.
pub(crate) async fn get_sampler_trace(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TraceQuery>,
) -> Result<Response, ServerError> {
    let Some(id) = query.id.filter(|id| !id.is_empty()) else {
        return Err(ServerError::NotFound("sampler-trace needs an id".into()));
    };
    let Some((meta, buffer)) = state.sampler_trace(&id) else {
        return Err(ServerError::NotFound(format!(
            "no sampler trace retained for {id}"
        )));
    };
    let buffer = buffer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let body = packet(&id, &meta, &buffer);
    Ok((
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store")],
        axum::Json(body),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::sampler::{RecordedSample, SampleObservation, Sampler};

    /// Verified upper bound on one serialized position's bytes. Six candidates
    /// (five shown plus a rank-outside selected) with large ids,
    /// shortest-roundtrip full-precision probabilities, and a 32-byte piece
    /// measure 451 on serde_json 1.0.151; the test below fails if the format
    /// ever grows past this bound, so the position budgets of this module
    /// stay honest.
    const WORST_CASE_BYTES_PER_POSITION: usize = 464;

    fn filled_handle(first: u32, positions: usize, limit: usize) -> CaptureHandle {
        let mut sampler = Sampler::greedy();
        let handle = sampler.arm_capture(limit);
        for n in 0..positions {
            sampler.record(&SampleObservation {
                selected: first + n as u32,
                selected_rank: 0,
                kept: 1,
                greedy: true,
                candidates: Vec::new(),
            });
        }
        handle
    }

    fn meta() -> TraceMeta {
        TraceMeta {
            model: "test-model".into(),
            temperature: 0.5,
            top_p: 0.9,
            top_k: 40,
        }
    }

    #[test]
    fn ring_evicts_oldest_beyond_the_reply_window() {
        let ring = SamplerTraceRing::default();
        for reply in 0..MAX_RETAINED_REPLIES + 2 {
            ring.register(&format!("r{reply}"), meta(), filled_handle(0, 100, 200));
        }
        assert!(ring.get("r0").is_none(), "older than the reply window");
        assert!(ring.get("r1").is_none());
        assert!(ring.get("r2").is_some(), "inside the reply window");
        assert!(
            ring.get(&format!("r{}", MAX_RETAINED_REPLIES + 1))
                .is_some()
        );
    }

    #[test]
    fn ring_replaces_a_same_id_registration() {
        let ring = SamplerTraceRing::default();
        ring.register("dup", meta(), filled_handle(0, 10, 100));
        ring.register("dup", meta(), filled_handle(100, 5, 100));
        let (_, handle) = ring.get("dup").expect("replaced entry");
        let buffer = handle.lock().expect("buffer lock");
        assert_eq!(buffer.positions.len(), 5);
        assert_eq!(buffer.positions[0].token, 100);
    }

    #[test]
    fn ring_enforces_the_global_position_budget_on_register() {
        let ring = SamplerTraceRing::default();
        // Entries at the full per-request size: the global budget fits
        // MAX_RETAINED_REPLIES of them, so registering one more evicts.
        for reply in 0..MAX_RETAINED_REPLIES {
            ring.register(
                &format!("full{reply}"),
                meta(),
                filled_handle(0, PER_REQUEST_POSITION_LIMIT, PER_REQUEST_POSITION_LIMIT),
            );
        }
        assert!(ring.get("full0").is_some());
        ring.register(
            "overflow",
            meta(),
            filled_handle(0, 1, PER_REQUEST_POSITION_LIMIT),
        );
        assert!(
            ring.get("full0").is_none(),
            "the global budget evicts oldest"
        );
        assert!(ring.get("overflow").is_some());
    }

    #[test]
    fn serialized_worst_case_position_stays_under_the_documented_bound() {
        // Six candidates (five shown plus a rank-outside selected), large ids,
        // full-precision probabilities, and a 32-byte piece with a max offset:
        // the measured serialized size must stay under
        // WORST_CASE_BYTES_PER_POSITION, or the memory budgets are lies.
        let mut sampler = Sampler::greedy();
        let handle = sampler.arm_capture(1);
        sampler.record(&SampleObservation {
            selected: 152_000,
            selected_rank: 9,
            kept: 512,
            greedy: false,
            candidates: vec![
                (151_996, 0.30000000000000004),
                (151_997, 0.20000000000000004),
                (151_998, 0.10000000000000002),
                (151_999, 0.05000000000000001),
                (151_995, 0.020000000000000004),
                (152_000, 0.33000000000000007),
            ],
        });
        {
            let mut buffer = handle.lock().expect("buffer lock");
            buffer.positions[0].piece = Some("x".repeat(32));
            buffer.positions[0].offset = Some(u32::MAX);
        }
        let buffer = handle.lock().expect("buffer lock");
        let body = packet("chatcmpl-x", &meta(), &buffer);
        let per_position = serde_json::to_string(&body["positions"][0]).expect("serialize");
        assert!(
            per_position.len() < WORST_CASE_BYTES_PER_POSITION,
            "serialized {} bytes, bound {WORST_CASE_BYTES_PER_POSITION}: {per_position}",
            per_position.len()
        );
    }

    #[test]
    fn packet_reports_truncation_counts_and_metadata() {
        let handle = filled_handle(0, 10, 10);
        {
            let mut buffer = handle.lock().expect("buffer lock");
            buffer.truncated = true;
        }
        let buffer = handle.lock().expect("buffer lock");
        let body = packet("r", &meta(), &buffer);
        assert_eq!(body["packet"], 2);
        assert_eq!(body["truncated"], true);
        assert_eq!(body["captured_positions"], 10);
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["sampling"]["temperature"], 0.5);
        assert_eq!(body["sampling"]["top_k"], 40);
        assert_eq!(body["capture_scope"], "decode");
        assert_eq!(body["positions"].as_array().expect("array").len(), 10);
        // Unreplayed positions serialize bare alignment fields: explicit
        // nulls, not absent keys, so a consumer always sees the same shape.
        assert_eq!(body["positions"][0]["offset"], serde_json::Value::Null);
        assert_eq!(body["positions"][0]["piece"], serde_json::Value::Null);
        assert_eq!(body["aligned_positions"], 0);
        assert_eq!(body["raw_byte_len"], 0);
        assert_eq!(body["alignment_error"], false);
        assert_eq!(body["trailer"], "");
    }

    #[test]
    fn aligner_attaches_pieces_offsets_and_survives_the_unicode_split() {
        // 'é' is ids [195, 169] on the byte-level fixture tokenizer: the first
        // id alone is an incomplete UTF-8 sequence, so its piece is empty and
        // the NEXT position carries the whole character — offsets must stay
        // monotone through that split.
        let mut sampler = Sampler::greedy();
        let buffer = sampler.arm_capture(8);
        for winner in [195_usize, 169, 97] {
            let mut row = vec![0.0_f32; 258];
            row[winner] = 1.0;
            sampler.sample(&row).expect("greedy sample");
        }
        let mut aligner = TraceAligner::new(testkit::bundle().tokenizer(), buffer.clone());
        aligner.on_batch(&[195, 169]);
        aligner.on_batch(&[97]);
        aligner.finish();
        let buffer = buffer.lock().expect("buffer lock");
        assert!(!buffer.alignment_error);
        let pieces: Vec<&str> = buffer
            .positions
            .iter()
            .map(|recorded| recorded.piece.as_deref().expect("aligned piece"))
            .collect();
        assert_eq!(pieces, ["", "\u{e9}", "a"]);
        assert_eq!(buffer.positions[0].offset, Some(0));
        assert_eq!(
            buffer.positions[1].offset,
            Some(0),
            "the empty piece advanced nothing"
        );
        assert_eq!(buffer.positions[2].offset, Some(2));
        assert_eq!(buffer.raw_len, 3);
        assert_eq!(buffer.trailer, "");
    }

    #[test]
    fn aligner_latches_on_token_mismatch_and_keeps_earlier_offsets() {
        // The replay feeds a token the sampler never recorded for that
        // position: alignment stops with the flag set, and the offsets
        // already written stay valid.
        let mut sampler = Sampler::greedy();
        let buffer = sampler.arm_capture(8);
        for _ in 0..2 {
            let mut row = vec![0.0_f32; 258];
            row[5] = 1.0;
            sampler.sample(&row).expect("greedy sample");
        }
        let mut aligner = TraceAligner::new(testkit::bundle().tokenizer(), buffer.clone());
        aligner.on_batch(&[5, 6]);
        aligner.on_batch(&[5]);
        aligner.finish();
        let buffer = buffer.lock().expect("buffer lock");
        assert!(buffer.alignment_error);
        let first = &buffer.positions[0];
        assert_eq!(first.piece.as_deref(), Some("\u{5}"));
        assert_eq!(first.offset, Some(0));
        assert_eq!(
            buffer.positions[1].piece, None,
            "alignment stopped at the break"
        );
        assert_eq!(buffer.raw_len, 1);
        assert_eq!(buffer.trailer, "", "a stopped aligner flushes nothing");
    }

    #[test]
    fn recorded_sample_round_trips_the_observation_fields() {
        let handle = filled_handle(7, 1, 8);
        let buffer = handle.lock().expect("buffer lock");
        let recorded: &RecordedSample = &buffer.positions[0];
        assert_eq!(recorded.token, 7);
        assert_eq!(recorded.kept, 1);
        assert!(recorded.greedy);
        assert_eq!(recorded.selected_prob, None);
    }

    // ---- Route e2e over the probe engine ---------------------------------------
    //
    // The probe engine's decode drives the REQUEST's sampler (it presents a
    // two-token row), so an armed request's capture holds real positions —
    // the honest end-to-end check of arm, record, ring, and route.

    use std::sync::Arc;

    use axum::http::{StatusCode, header};
    use serde_json::{Value, json};

    use super::super::router::router;
    use super::super::state::AppState;
    use super::super::testkit;

    fn app(sample_trace: bool) -> axum::Router {
        let probes = testkit::probes(std::time::Duration::ZERO);
        let entry = testkit::entry("e1", "model-a", &probes.engine, 100_000);
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_sample_trace(sample_trace);
        router(Arc::new(state))
    }

    #[tokio::test]
    async fn an_armed_request_leaves_a_readable_packet_and_an_unchanged_reply() {
        let mut replies: Vec<Value> = Vec::new();
        for trace in [false, true] {
            let app = app(true);
            let body = testkit::post_json(
                "/v1/chat/completions",
                &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 3, "x-sample-trace": trace}),
            );
            let (status, _headers, bytes) = testkit::send(&app, body).await;
            assert_eq!(status, StatusCode::OK);
            let reply = testkit::json_of(&bytes);
            assert_eq!(reply["choices"][0]["finish_reason"], "length");
            replies.push(reply.clone());
            if !trace {
                continue;
            }
            // The packet: three greedy positions from the probe's two-token
            // row, no probabilities, this reply's id and model.
            let get = testkit::get(
                &format!("/v1/sampler-trace?id={}", reply["id"].as_str().expect("id")),
                Some(testkit::API_KEY),
            );
            let (status, headers, bytes) = testkit::send(&app, get).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                headers
                    .get(header::CACHE_CONTROL)
                    .and_then(|v| v.to_str().ok()),
                Some("no-store")
            );
            let packet = testkit::json_of(&bytes);
            assert_eq!(packet["packet"], 2);
            assert_eq!(packet["reply_id"], reply["id"]);
            assert_eq!(packet["model"], "model-a");
            assert_eq!(packet["sampling"]["temperature"].as_f64(), Some(0.0));
            assert_eq!(packet["truncated"], false);
            assert_eq!(packet["captured_positions"], 3);
            let positions = packet["positions"].as_array().expect("positions");
            assert_eq!(positions.len(), 3);
            for position in positions {
                assert_eq!(position["token_id"], 1, "greedy argmax of [0.0, 1.0]");
                assert_eq!(position["greedy"], true);
                assert_eq!(position["selected_prob"], Value::Null);
                assert_eq!(
                    position["candidates"].as_array().expect("candidates").len(),
                    0
                );
            }
            // Alignment: every position carries its piece, the pieces
            // concatenate to exactly the reply's content text, and the
            // offsets walk it byte by byte — the guarantee the GUI highlights
            // against.
            assert_eq!(packet["aligned_positions"], 3);
            assert_eq!(packet["alignment_error"], false);
            let content = reply["choices"][0]["message"]["content"]
                .as_str()
                .expect("content")
                .to_string();
            let mut joined = String::new();
            let mut offsets = Vec::new();
            for position in positions {
                joined.push_str(position["piece"].as_str().expect("aligned piece"));
                offsets.push(position["offset"].as_u64());
            }
            assert_eq!(joined, content, "pieces concatenate to the reply text");
            assert_eq!(offsets, vec![Some(0), Some(1), Some(2)]);
            assert_eq!(packet["raw_byte_len"], content.len());
            assert_eq!(packet["trailer"], "");
        }
        // The requests can cross a wall-clock second. Each must carry a timestamp.
        for reply in &mut replies {
            let created = reply
                .as_object_mut()
                .expect("reply object")
                .remove("created");
            assert!(
                created.and_then(|value| value.as_u64()).is_some(),
                "missing or invalid creation timestamp"
            );
        }
        assert_eq!(replies[0], replies[1], "arming capture changed the reply");
    }

    #[test]
    fn aligner_latches_alignment_error_on_decode_failure() {
        // An id outside the vocabulary fails the decoder's replay: the
        // contract says alignment_error latches — not just missing offsets.
        let mut sampler = Sampler::greedy();
        let buffer = sampler.arm_capture(8);
        for _ in 0..2 {
            let mut row = vec![0.0_f32; 258];
            row[5] = 1.0;
            sampler.sample(&row).expect("greedy sample");
        }
        let mut aligner = TraceAligner::new(testkit::bundle().tokenizer(), buffer.clone());
        aligner.on_batch(&[5]);
        aligner.on_batch(&[9999]);
        aligner.finish();
        let buffer = buffer.lock().expect("buffer lock");
        assert!(buffer.alignment_error, "decode failure must latch");
        assert_eq!(buffer.positions[0].offset, Some(0), "earlier offsets stay");
        assert_eq!(buffer.positions[1].offset, None);
        assert_eq!(buffer.trailer, "", "a stopped aligner flushes nothing");
    }

    #[test]
    fn truncated_capture_stops_the_replay_cleanly() {
        // The buffer stopped recording at its limit while generation ran on:
        // the replay's extra pieces are expected, so alignment stops without
        // the error flag — truncation already says what happened.
        let mut sampler = Sampler::greedy();
        let buffer = sampler.arm_capture(2);
        for _ in 0..3 {
            let mut row = vec![0.0_f32; 258];
            row[5] = 1.0;
            sampler.sample(&row).expect("greedy sample");
        }
        let mut aligner = TraceAligner::new(testkit::bundle().tokenizer(), buffer.clone());
        aligner.on_batch(&[5, 5, 5]);
        aligner.finish();
        let buffer = buffer.lock().expect("buffer lock");
        assert!(buffer.truncated);
        assert!(!buffer.alignment_error, "clean stop, not a disagreement");
        assert_eq!(buffer.positions.len(), 2);
        assert_eq!(buffer.positions[0].offset, Some(0));
        assert_eq!(buffer.positions[1].offset, Some(1));
    }

    #[tokio::test]
    async fn capture_is_refused_when_the_process_does_not_permit_it() {
        let app = app(false);
        let body = testkit::post_json(
            "/v1/chat/completions",
            &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 1, "x-sample-trace": true}),
        );
        let (status, _headers, bytes) = testkit::send(&app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let error = testkit::json_of(&bytes);
        let text = error.to_string();
        assert!(text.contains("sample_trace"), "{text}");
    }

    #[tokio::test]
    async fn an_unarmed_request_registers_nothing_and_unknown_ids_are_not_found() {
        let app = app(true);
        let body = testkit::post_json(
            "/v1/chat/completions",
            &json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 1}),
        );
        let (status, _headers, bytes) = testkit::send(&app, body).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let id = reply["id"].as_str().expect("id");
        let unarmed = testkit::get(
            &format!("/v1/sampler-trace?id={id}"),
            Some(testkit::API_KEY),
        );
        let (status, _h, _b) = testkit::send(&app, unarmed).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "no field: no trace");
        let unknown = testkit::get("/v1/sampler-trace?id=chatcmpl-nope", Some(testkit::API_KEY));
        let (status, _h, _b) = testkit::send(&app, unknown).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let bare = testkit::get("/v1/sampler-trace", Some(testkit::API_KEY));
        let (status, _h, _b) = testkit::send(&app, bare).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "missing id is explicit");
    }
}
