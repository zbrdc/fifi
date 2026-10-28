//! Test-only capture of the checked, original-rank router IDs.
use crate::EngError;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

type Result<T> = std::result::Result<T, EngError>;
const BYTE_LIMIT: usize = 32 << 20;
thread_local! {
    static TRACE: RefCell<Slot> = const { RefCell::new(Slot { epoch: 0, state: None }) };
}
struct Slot {
    epoch: u64,
    state: Option<State>,
}
struct State {
    writer: BufWriter<File>,
    bytes: usize,
    limit: usize,
    request: String,
    expected: u64,
    prompt: usize,
    output_tokens: usize,
    failure: Option<String>,
    phase: Option<&'static str>,
    pass: Option<Pass>,
    layer: Option<usize>,
    passes: usize,
    end: usize,
    prefill_rows: usize,
    decode_rows: usize,
    counts: [[usize; 2]; 49],
}
struct Pass {
    position: usize,
    rows: usize,
    seen: u64,
}
#[derive(Clone, Copy)]
enum Kind {
    Phase,
    Pass,
    Layer,
}
#[must_use]
pub(crate) struct Guard {
    epoch: Option<u64>,
    kind: Kind,
    _thread: PhantomData<Rc<()>>,
}
#[must_use]
pub(crate) struct Capture {
    epoch: u64,
    _thread: PhantomData<Rc<()>>,
}
pub(crate) struct Stopped(State);

fn bad(message: impl std::fmt::Display) -> EngError {
    EngError::BadRequest(format!("Flint route trace: {message}"))
}
fn active(action: impl FnOnce(&mut State) -> Result<()>) -> Result<Option<u64>> {
    TRACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let epoch = slot.epoch;
        let Some(state) = slot.state.as_mut() else {
            return Ok(None);
        };
        if let Some(error) = &state.failure {
            return Err(bad(error));
        }
        if let Err(error) = action(state) {
            state.failure = Some(format!("{error:?}"));
            return Err(error);
        }
        Ok(Some(epoch))
    })
}
fn scope(kind: Kind, action: impl FnOnce(&mut State) -> Result<()>) -> Result<Guard> {
    Ok(Guard {
        epoch: active(action)?,
        kind,
        _thread: PhantomData,
    })
}
pub(crate) fn phase(phase: &'static str) -> Result<Guard> {
    scope(Kind::Phase, |state| {
        if state.phase.is_some() || !matches!(phase, "prefill" | "decode") {
            return Err(bad("missing or nested API phase"));
        }
        if phase == "prefill" && state.decode_rows != 0 {
            return Err(bad("stale prefill phase"));
        }
        state.phase = Some(phase);
        Ok(())
    })
}
pub(crate) fn pass(position: usize, rows: usize) -> Result<Guard> {
    scope(Kind::Pass, |state| {
        if state.phase.is_none()
            || state.pass.is_some()
            || state.layer.is_some()
            || rows == 0
            || rows > 8
            || position != state.end
        {
            return Err(bad("missing, nested or noncontiguous pass"));
        }
        if state.phase == Some("decode") && rows != 1 {
            return Err(bad("decode must feed one row"));
        }
        state.pass = Some(Pass {
            position,
            rows,
            seen: 0,
        });
        Ok(())
    })
}
pub(crate) fn layer(index: usize) -> Result<Guard> {
    scope(Kind::Layer, |state| {
        if state.pass.is_none() || state.layer.is_some() || index >= 49 {
            return Err(bad("missing, nested or invalid layer"));
        }
        state.layer = Some(index);
        Ok(())
    })
}
impl Drop for Guard {
    fn drop(&mut self) {
        let Some(epoch) = self.epoch else { return };
        TRACE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let current = slot.epoch;
            let Some(state) = slot.state.as_mut() else {
                return;
            };
            if current != epoch {
                state.failure = Some("stale guard from another capture".into());
                return;
            }
            if std::thread::panicking() {
                state
                    .failure
                    .get_or_insert("scope unwound before completion".into());
            }
            let valid = match self.kind {
                Kind::Layer => state.layer.take().is_some(),
                Kind::Phase => state.phase.take().is_some() && state.pass.is_none(),
                Kind::Pass => state.close_pass(),
            };
            if !valid {
                state
                    .failure
                    .get_or_insert("incomplete or misordered scope".into());
            }
        });
    }
}
impl State {
    fn write(&mut self, record: &Value) -> Result<()> {
        let bytes = serde_json::to_vec(record).map_err(bad)?;
        let next = self
            .bytes
            .checked_add(bytes.len())
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| bad("serialized byte cap exceeded"))?;
        self.writer
            .write_all(&bytes)
            .and_then(|()| self.writer.write_all(b"\n"))
            .map_err(bad)?;
        self.bytes = next;
        Ok(())
    }
    fn close_pass(&mut self) -> bool {
        let Some(pass) = self.pass.take() else {
            return false;
        };
        if self.layer.is_some() || pass.seen != self.expected {
            return false;
        }
        let Some(end) = pass.position.checked_add(pass.rows) else {
            return false;
        };
        self.end = end;
        self.passes += 1;
        match self.phase {
            Some("prefill") => self.prefill_rows += pass.rows,
            Some("decode") => self.decode_rows += pass.rows,
            _ => return false,
        }
        true
    }
    fn check_finished(&self) -> Result<()> {
        if let Some(error) = &self.failure {
            return Err(bad(error));
        }
        if self.phase.is_some()
            || self.pass.is_some()
            || self.layer.is_some()
            || self.prefill_rows != self.prompt - 1
            || self.decode_rows != self.output_tokens
            || self.end != self.prompt + self.output_tokens - 1
        {
            return Err(bad("incomplete request or prompt-tail accounting"));
        }
        for (layer, &[records, rows]) in self.counts.iter().enumerate() {
            let expected = self.expected & (1 << layer) != 0;
            if (expected && (records != self.passes || rows != self.end))
                || (!expected && (records != 0 || rows != 0))
            {
                return Err(bad("incomplete layer counts"));
            }
        }
        Ok(())
    }
}
pub(crate) fn record(ids: &[i32], rows: usize, experts: usize) -> Result<()> {
    active(|state| {
        let layer = state
            .layer
            .ok_or_else(|| bad("router has no layer context"))?;
        let pass = state
            .pass
            .as_ref()
            .ok_or_else(|| bad("router has no pass context"))?;
        if experts != 256
            || rows != pass.rows
            || ids.len() != rows * 6
            || state.expected & (1 << layer) == 0
            || pass.seen & (1 << layer) != 0
        {
            return Err(bad("router geometry or duplicate layer"));
        }
        for row in ids.as_chunks::<6>().0 {
            for (rank, &id) in row.iter().enumerate() {
                if !(0..256).contains(&id) || row[..rank].contains(&id) {
                    return Err(bad("router IDs must be six distinct IDs in 0..256"));
                }
            }
        }
        let line = json!({"kind":"route", "request_id":state.request,
            "phase":state.phase, "pass_index":state.passes, "actual_layer":layer,
            "start_position":pass.position, "rows":rows, "top_k":6,
            "ids":ids.as_chunks::<6>().0});
        state.write(&line)?;
        state.pass.as_mut().unwrap().seen |= 1 << layer;
        state.counts[layer][0] += 1;
        state.counts[layer][1] += rows;
        Ok(())
    })
    .map(|_| ())
}
impl Capture {
    pub(crate) fn begin(
        path: &Path,
        mut header: Value,
        layers: &[usize],
        prompt: usize,
    ) -> Result<Self> {
        if !path.is_absolute()
            || (!(2..=4000).contains(&prompt) && !matches!(prompt, 11444 | 16384))
            || layers.len() != 47
        {
            return Err(bad(
                "absolute output, bounded prompt and 47 MoE layers required",
            ));
        }
        let (output_tokens, limit) = if matches!(prompt, 11444 | 16384) {
            (256, 64 << 20)
        } else {
            (64, BYTE_LIMIT)
        };
        if header["max_tokens"] != output_tokens || header["byte_cap"] != limit {
            return Err(bad(
                "header output count or byte cap differs from capture bounds",
            ));
        }
        let mut expected = 0u64;
        for &layer in layers {
            if layer >= 49 || expected & (1 << layer) != 0 {
                return Err(bad("invalid MoE layer list"));
            }
            expected |= 1 << layer;
        }
        let request = header["request_id"]
            .as_str()
            .ok_or_else(|| bad("missing request ID"))?
            .to_owned();
        header["kind"] = "header".into();
        header["schema"] = "flint-routes/1".into();
        header["complete"] = false.into();
        header["moe_layers"] = json!(layers);
        TRACE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(state) = slot.state.as_mut() {
                state.failure = Some("nested capture".into());
                return Err(bad("nested capture"));
            }
            let epoch = slot
                .epoch
                .checked_add(1)
                .ok_or_else(|| bad("capture epoch overflow"))?;
            let writer = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .map_err(bad)?,
            );
            let mut state = State {
                writer,
                bytes: 0,
                limit,
                request,
                expected,
                prompt,
                output_tokens,
                failure: None,
                phase: None,
                pass: None,
                layer: None,
                passes: 0,
                end: 0,
                prefill_rows: 0,
                decode_rows: 0,
                counts: [[0; 2]; 49],
            };
            state.write(&header)?;
            slot.epoch = epoch;
            slot.state = Some(state);
            Ok(Self {
                epoch,
                _thread: PhantomData,
            })
        })
    }
    pub(crate) fn stop(self) -> Result<Stopped> {
        TRACE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.epoch != self.epoch {
                return Err(bad("stale capture"));
            }
            let state = slot.state.take().ok_or_else(|| bad("missing capture"))?;
            state.check_finished()?;
            Ok(Stopped(state))
        })
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        TRACE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.epoch == self.epoch {
                slot.state.take();
            }
        });
    }
}
impl Stopped {
    pub(crate) fn complete(mut self, mut footer: Value, cleanup: Result<usize>) -> Result<()> {
        if cleanup? == 0 {
            return Err(bad("no healthy CUDA device after cleanup"));
        }
        self.0.check_finished()?;
        if footer["generated_ids"].as_array().map(Vec::len) != Some(self.0.output_tokens)
            || footer["control_equal"] != true
        {
            return Err(bad("missing expected-token control equality"));
        }
        footer["kind"] = "footer".into();
        footer["complete"] = true.into();
        footer["dropped_records"] = 0.into();
        footer["passes"] = self.0.passes.into();
        footer["fed_rows"] = self.0.end.into();
        footer["prefill_rows"] = self.0.prefill_rows.into();
        footer["decode_rows"] = self.0.decode_rows.into();
        footer["per_layer_records_rows"] = json!(self.0.counts.as_slice());
        self.0.write(&footer)?;
        self.0.writer.flush().map_err(bad)?;
        self.0.writer.get_ref().sync_all().map_err(bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const IDS: [i32; 6] = [17, 3, 255, 0, 42, 9];
    fn start(dir: &tempfile::TempDir) -> Capture {
        Capture::begin(
            &dir.path().join("trace.jsonl"),
            json!({"request_id":"cpu","max_tokens":64,"byte_cap":32<<20}),
            &(2..49).collect::<Vec<_>>(),
            2,
        )
        .unwrap()
    }
    fn feed(position: usize, rows: usize) {
        let _pass = pass(position, rows).unwrap();
        let ids: Vec<_> = (0..rows)
            .flat_map(|row| IDS.map(|id| (id + i32::try_from(row).unwrap()) % 256))
            .collect();
        for index in 2..49 {
            let _layer = layer(index).unwrap();
            record(&ids, rows, 256).unwrap();
        }
    }
    fn finish_request() {
        {
            let _phase = phase("prefill").unwrap();
            feed(0, 1);
        }
        let _phase = phase("decode").unwrap();
        for position in 1..65 {
            feed(position, 1);
        }
    }
    #[test]
    fn inactive_hook_and_guards_do_not_validate_or_create_state() {
        let _phase = phase("invalid").unwrap();
        let _pass = pass(usize::MAX, usize::MAX).unwrap();
        let _layer = layer(usize::MAX).unwrap();
        record(&[-1], 0, 0).unwrap();
        TRACE.with(|slot| assert!(slot.borrow().state.is_none()));
    }
    #[test]
    fn exact_rank_rows_and_prompt_tail_complete_after_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let capture = Capture::begin(
            &dir.path().join("trace.jsonl"),
            json!({"request_id":"cpu","max_tokens":64,"byte_cap":32<<20}),
            &(2..49).collect::<Vec<_>>(),
            16,
        )
        .unwrap();
        {
            let _phase = phase("prefill").unwrap();
            feed(0, 8);
            feed(8, 7);
        }
        {
            let _phase = phase("decode").unwrap();
            for position in 15..79 {
                feed(position, 1);
            }
        }
        capture
            .stop()
            .unwrap()
            .complete(
                json!({"generated_ids":vec![1;64],"control_equal":true}),
                Ok(1),
            )
            .unwrap();
        let text = std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap();
        let records: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let expected = [
            [17, 3, 255, 0, 42, 9],
            [18, 4, 0, 1, 43, 10],
            [19, 5, 1, 2, 44, 11],
            [20, 6, 2, 3, 45, 12],
            [21, 7, 3, 4, 46, 13],
            [22, 8, 4, 5, 47, 14],
            [23, 9, 5, 6, 48, 15],
            [24, 10, 6, 7, 49, 16],
        ];
        assert_eq!(records[1]["ids"], json!(expected));
        assert_eq!(records[1]["phase"], "prefill");
        assert_eq!(records[1]["rows"], 8);
        assert_eq!(records[48]["ids"], json!(&expected[..7]));
        assert_eq!(records[48]["start_position"], 8);
        assert_eq!(records[48]["rows"], 7);
        let footer = records.last().unwrap();
        assert_eq!(footer["fed_rows"], 79);
        assert_eq!(footer["prefill_rows"], 15);
        assert_eq!(footer["decode_rows"], 64);
        assert_eq!(footer["passes"], 66);
        assert_eq!(footer["per_layer_records_rows"][2], json!([66, 79]));
        assert_eq!(footer["complete"], true);
    }
    #[test]
    fn retained_16k_256_geometry_completes_with_bounded_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let capture = Capture::begin(
            &path,
            json!({"request_id":"cpu-16k","max_tokens":256,"byte_cap":64<<20}),
            &(2..49).collect::<Vec<_>>(),
            16384,
        )
        .expect("approved 16K/256 capture geometry must be accepted");
        TRACE.with(|slot| assert_eq!(slot.borrow().state.as_ref().unwrap().limit, 64 << 20));
        {
            let _phase = phase("prefill").unwrap();
            for position in (0..16383).step_by(8) {
                feed(position, (16383 - position).min(8));
            }
        }
        {
            let _phase = phase("decode").unwrap();
            for position in 16383..16639 {
                feed(position, 1);
            }
        }
        capture
            .stop()
            .unwrap()
            .complete(
                json!({"generated_ids":vec![1;256],"control_equal":true}),
                Ok(1),
            )
            .unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text.lines().count(), 108290);
        assert!(text.len() < 64 << 20);
        let footer: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        assert_eq!(footer["prefill_rows"], 16383);
        assert_eq!(footer["decode_rows"], 256);
        assert_eq!(footer["fed_rows"], 16639);
        assert_eq!(footer["passes"], 2304);
        assert_eq!(footer["generated_ids"].as_array().unwrap().len(), 256);
        for index in 2..49 {
            assert_eq!(
                footer["per_layer_records_rows"][index],
                json!([2304, 16639])
            );
        }
        assert_eq!(footer["complete"], true);
    }
    #[test]
    fn retained_h2_256_geometry_completes_with_bounded_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let capture = Capture::begin(
            &path,
            json!({"request_id":"cpu-h2","max_tokens":256,"byte_cap":64<<20}),
            &(2..49).collect::<Vec<_>>(),
            11444,
        )
        .expect("approved H2/256 capture geometry must be accepted");
        TRACE.with(|slot| assert_eq!(slot.borrow().state.as_ref().unwrap().limit, 64 << 20));
        {
            let _phase = phase("prefill").unwrap();
            for position in (0..11443).step_by(8) {
                feed(position, (11443 - position).min(8));
            }
        }
        {
            let _phase = phase("decode").unwrap();
            for position in 11443..11699 {
                feed(position, 1);
            }
        }
        capture
            .stop()
            .unwrap()
            .complete(
                json!({"generated_ids":vec![1;256],"control_equal":true}),
                Ok(1),
            )
            .unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text.lines().count(), 79291);
        assert!(text.len() < 64 << 20);
        let footer: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        assert_eq!(footer["prefill_rows"], 11443);
        assert_eq!(footer["decode_rows"], 256);
        assert_eq!(footer["fed_rows"], 11699);
        assert_eq!(footer["passes"], 1687);
        assert_eq!(footer["generated_ids"].as_array().unwrap().len(), 256);
        for index in 2..49 {
            assert_eq!(
                footer["per_layer_records_rows"][index],
                json!([1687, 11699])
            );
        }
        assert_eq!(footer["complete"], true);
    }
    #[test]
    fn capture_bounds_and_header_counts_must_match() {
        for (prompt, outputs, cap) in [
            (1, 64, 32 << 20),
            (4001, 64, 32 << 20),
            (11443, 256, 64 << 20),
            (11445, 256, 64 << 20),
            (11444, 64, 64 << 20),
            (11444, 256, 32 << 20),
            (16383, 256, 64 << 20),
            (16385, 256, 64 << 20),
            (16384, 64, 64 << 20),
            (16384, 256, 32 << 20),
            (4000, 256, 64 << 20),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("trace.jsonl");
            assert!(
                Capture::begin(
                    &path,
                    json!({"request_id":"bounds","max_tokens":outputs,"byte_cap":cap}),
                    &(2..49).collect::<Vec<_>>(),
                    prompt
                )
                .is_err()
            );
            assert!(!path.exists());
        }
        let dir = tempfile::tempdir().unwrap();
        let capture = Capture::begin(
            &dir.path().join("trace.jsonl"),
            json!({"request_id":"legacy-max","max_tokens":64,"byte_cap":32<<20}),
            &(2..49).collect::<Vec<_>>(),
            4000,
        )
        .unwrap();
        drop(capture);
    }
    #[test]
    fn short_16k_decode_and_short_footer_never_complete() {
        for outputs in [64, 256] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("trace.jsonl");
            let capture = Capture::begin(
                &path,
                json!({"request_id":"short-16k","max_tokens":256,"byte_cap":64<<20}),
                &(2..49).collect::<Vec<_>>(),
                16384,
            )
            .unwrap();
            {
                let _phase = phase("prefill").unwrap();
                for position in (0..16383).step_by(8) {
                    feed(position, (16383 - position).min(8));
                }
            }
            {
                let _phase = phase("decode").unwrap();
                for position in 16383..16383 + outputs {
                    feed(position, 1);
                }
            }
            let stopped = capture.stop();
            if outputs == 64 {
                assert!(stopped.is_err());
            } else {
                assert!(
                    stopped
                        .unwrap()
                        .complete(
                            json!({"generated_ids":vec![1;64],"control_equal":true}),
                            Ok(1)
                        )
                        .is_err()
                );
            }
            assert!(
                !std::fs::read_to_string(path)
                    .unwrap()
                    .contains("\"complete\":true")
            );
        }
    }
    #[test]
    fn bad_ids_lengths_and_duplicates_latch_failure() {
        for ids in [
            vec![-1, 3, 4, 5, 6, 7],
            vec![256, 3, 4, 5, 6, 7],
            vec![1; 6],
            vec![1; 5],
        ] {
            let dir = tempfile::tempdir().unwrap();
            let capture = start(&dir);
            {
                let _phase = phase("prefill").unwrap();
                let _pass = pass(0, 1).unwrap();
                let _layer = layer(2).unwrap();
                assert!(record(&ids, 1, 256).is_err());
            }
            assert!(capture.stop().is_err());
        }
    }
    #[test]
    fn missing_nested_stale_contexts_and_unwind_stay_incomplete() {
        for case in 0..9 {
            let dir = tempfile::tempdir().unwrap();
            let capture = start(&dir);
            match case {
                0 => {
                    assert!(record(&IDS, 1, 256).is_err());
                }
                1 => {
                    let _phase = phase("prefill").unwrap();
                    assert!(phase("decode").is_err());
                }
                2 => {
                    let _phase = phase("prefill").unwrap();
                    let _pass = pass(0, 1).unwrap();
                    assert!(pass(0, 1).is_err());
                }
                3 => {
                    let _phase = phase("prefill").unwrap();
                    assert!(pass(1, 1).is_err());
                }
                4 => {
                    assert!(pass(0, 1).is_err());
                }
                5 => {
                    assert!(layer(2).is_err());
                }
                6 => {
                    let _phase = phase("prefill").unwrap();
                    let _pass = pass(0, 1).unwrap();
                    let _layer = layer(2).unwrap();
                    assert!(layer(3).is_err());
                }
                7 => {
                    let _phase = phase("prefill").unwrap();
                    let _pass = pass(0, 1).unwrap();
                    let _layer = layer(2).unwrap();
                    record(&IDS, 1, 256).unwrap();
                    assert!(record(&IDS, 1, 256).is_err());
                }
                _ => {
                    let result = std::panic::catch_unwind(|| {
                        let _phase = phase("prefill").unwrap();
                        let _pass = pass(0, 1).unwrap();
                        let _layer = layer(2).unwrap();
                        panic!("injected unwind");
                    });
                    assert!(result.is_err());
                }
            }
            assert!(capture.stop().is_err());
            let text = std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap();
            assert!(!text.contains("\"complete\":true"));
        }
    }
    #[test]
    fn byte_cap_io_and_cleanup_failure_never_write_complete_footer() {
        for case in 0..6 {
            let dir = tempfile::tempdir().unwrap();
            let capture = start(&dir);
            if case < 2 {
                TRACE.with(|slot| {
                    let mut slot = slot.borrow_mut();
                    let state = slot.state.as_mut().unwrap();
                    if case == 0 {
                        state.limit = state.bytes;
                    } else {
                        state.writer = BufWriter::with_capacity(
                            1,
                            File::open(dir.path().join("trace.jsonl")).unwrap(),
                        );
                    }
                });
                {
                    let _phase = phase("prefill").unwrap();
                    let _pass = pass(0, 1).unwrap();
                    let _layer = layer(2).unwrap();
                    assert!(record(&IDS, 1, 256).is_err());
                }
                assert!(capture.stop().is_err());
            } else {
                finish_request();
                let mut stopped = capture.stop().unwrap();
                let mut footer = json!({"generated_ids":vec![1;64],"control_equal":true});
                let cleanup = match case {
                    2 => Err(bad("cleanup")),
                    3 => {
                        stopped.0.limit = stopped.0.bytes;
                        Ok(1)
                    }
                    4 => {
                        footer["control_equal"] = false.into();
                        Ok(1)
                    }
                    _ => Ok(0),
                };
                assert!(stopped.complete(footer, cleanup).is_err());
            }
            let text = std::fs::read_to_string(dir.path().join("trace.jsonl")).unwrap();
            assert!(!text.contains("\"complete\":true"));
        }
    }
    #[test]
    fn output_is_absolute_create_new_and_old_guards_cannot_enter_new_capture() {
        assert!(
            Capture::begin(
                Path::new("relative"),
                json!({"request_id":"x","max_tokens":64,"byte_cap":32<<20}),
                &(2..49).collect::<Vec<_>>(),
                2
            )
            .is_err()
        );
        let dir = tempfile::tempdir().unwrap();
        let capture = start(&dir);
        let stale = phase("prefill").unwrap();
        drop(capture);
        assert!(
            Capture::begin(
                &dir.path().join("trace.jsonl"),
                json!({"request_id":"x","max_tokens":64,"byte_cap":32<<20}),
                &(2..49).collect::<Vec<_>>(),
                2
            )
            .is_err()
        );
        let next = tempfile::tempdir().unwrap();
        let capture = start(&next);
        drop(stale);
        assert!(phase("prefill").is_err());
        assert!(capture.stop().is_err());
    }
}
